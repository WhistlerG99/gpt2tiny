import os
import re
import math
import json
import shutil
import textwrap
import tempfile
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Optional, Tuple
import mlflow

from gpt2tiny.dataset import SFTDataset 
from gpt2tiny.callbacks import LogBestCkptAndPyfuncToMLflow

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import (
    LinearLR,
    CosineAnnealingLR,
    SequentialLR,
)
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
from pytorch_lightning.loggers import MLFlowLogger
from transformers import AutoTokenizer, GPT2LMHeadModel

BASE_DIR = "/teamspace/studios/this_studio/gpt2tiny"
DATA_CACHE_DIR = Path(BASE_DIR) / "data"

# mlflow_tracking_uri = "https://5000-01kgr6z0qq5h0srek4vj1jq4pb.cloudspaces.litng.ai"
# os.environ["MLFLOW_ARTIFACT_URI"] = "file:///teamspace/s3_folders/mlflow-job-artifacts"
# os.environ["MLFLOW_TRACKING_URI"] = mlflow_tracking_uri



MODEL_NAME = "gpt2"
DATA_DIR = DATA_CACHE_DIR / "TinyStories_custom_prompts_w_completions_sft_huggingface_gpt2_v1"

BATCH_SIZE = 8
NUM_WORKERS = 4
LR = 3e-5
WARMUP_RATIO=0.05
WEIGHT_DECAY = 0.01
MAX_STEPS = 20000
VAL_BATCHES = 50
VAL_CHECK_INTERVAL=100
LOG_EVERY_N_STEPS=100
GRAD_ACC_STEPS=8
GRAD_CLIP=1.0

EXPERIMENT_NAME = "gpt2_sft_tinystories_lightning"
RUN_NAME = "gpt2_sft_run"

CHECKPOINT_DIR = "./checkpoints"
MLRUNS_DIR = "./mlruns"

# Supply prompts here later
GENERATION_PROMPTS = [
    "Tell a story. In the story, work in the adjective 'jolly' and ensure "
    "that 'label' is used as a noun. Also ensure that plant an earlier detail "
    "that becomes important later and there should be a clear tension or struggle in the story.",
    "Create a short fictional story. The story should also satisfy the following: "
    "the narrative should have at least one spoken conversation, the story has some "
    "form of conflict in it, and the story should foreshadow a later event or reveal.",
    "Write a brief tale. Make sure the narrative make sure 'great' appears as an "
    "adjective, have 'succeed' function as a verb, and make sure 'animal' appears "
    "as a noun. Also ensure that the story should include an unexpected turn and "
    "plant an earlier detail that becomes important later.",
    "Create a short fictional story. In the story, ensure that 'castle' is used as a noun, "
    "ensure that 'step' is used as a noun, work in the verb 'prepare', and ensure that "
    "'rhythm' is used as a noun. The story should also satisfy the following: the story "
    "has a bad ending.",
    "Write an original story. Make it about an outlaw who gets invited to a secret "
    "meeting in a sunken chapel. The story should work in the noun 'mixer', ensure "
    "that 'gloomy' is used as an adjective, and work in the adjective 'different'.",
]


# Generation settings used whenever validation runs
GEN_MAX_NEW_TOKENS = 128
GEN_TEMPERATURE = 0.8
GEN_TOP_K = 50
GEN_TOP_P = 0.95

# Precision / hardware
MATMUL_PRECISION = "medium"   # "medium" or "high"
torch.set_float32_matmul_precision(MATMUL_PRECISION)


# ============================================================
# DataModule
# ============================================================
class SFTDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: Path,
        pad_token_id: int,
        batch_size: int = 64,
        num_workers: int = 1,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pad_token_id = pad_token_id

    def train_dataloader(self):
        return DataLoader(
            SFTDataset(
                split="train",
                data_dir=self.data_dir,
                weights="Balanced",
            ),
            collate_fn=lambda batch: SFTDataset.collator(batch, pad_id=self.pad_token_id),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    def val_dataloader(self):
        return DataLoader(
            SFTDataset(
                split="validation",
                data_dir=self.data_dir,
                weights="Balanced",
            ),
            collate_fn=lambda batch: SFTDataset.collator(batch, pad_id=self.pad_token_id),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
    
# ============================================================
# LightningModule
# ============================================================
class GPT2SFTModule(pl.LightningModule):
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        model_name: str = "gpt2",
        lr: float = 3e-5,
        warmup_ratio: float = 0.0,
        weight_decay: float = 0.01,
        generation_prompts: Optional[List[str]] = None,
        generation_max_new_tokens: int = 248,
        generation_temperature: float = 0.8,
        generation_top_k: int = 50,
        generation_top_p: float = 0.95,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["generation_prompts"])
        self.tokenizer = tokenizer
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.generation_prompts = generation_prompts or []

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        
        # --- total steps (Lightning-safe) ---
        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = int(self.hparams.warmup_ratio * total_steps)

        # --- Warmup: near 0 → base LR ---
        warmup = LinearLR(
            optimizer,
            start_factor=1e-8,
            end_factor=1.0,
            total_iters=warmup_steps,
        )

        # --- Cosine decay ---
        cosine = CosineAnnealingLR(
            optimizer,
            T_max=max(1, total_steps - warmup_steps),
            eta_min=0.0,
        )

        # --- Combine ---
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[warmup_steps],
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",   # critical for warmup
                "frequency": 1,
            },
        }

    def forward(self, input_ids, attention_mask=None):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )

    def _build_labels(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        question_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """
        Labels:
          - ignore prompt tokens
          - ignore padding
          - keep answer/completion tokens only
        """
        B, T = input_ids.shape
        labels = input_ids.clone()

        # ignore padding
        labels = labels.masked_fill(attention_mask == 0, -100)

        # ignore prompt tokens
        positions = torch.arange(T, device=input_ids.device).unsqueeze(0)  # (1, T)
        prompt_mask = positions < question_lengths.unsqueeze(1)            # (B, T)
        labels = labels.masked_fill(prompt_mask, -100)

        return labels

    def _loss_and_entropy(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        question_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            loss: scalar CE loss over completion tokens only
            entropy: scalar mean token entropy over completion tokens only
        """
        outputs = self(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        logits = outputs.logits  # (B, T, V)

        labels = self._build_labels(
            input_ids=input_ids,
            attention_mask=attention_mask,
            question_lengths=question_lengths,
        )

        shift_logits = logits[:, :-1, :].contiguous()   # predict token t+1 from token t
        shift_labels = labels[:, 1:].contiguous()

        loss = nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )

        # Entropy on supervised positions only
        with torch.no_grad():
            probs = torch.softmax(shift_logits, dim=-1)
            log_probs = torch.log_softmax(shift_logits, dim=-1)
            token_entropy = -(probs * log_probs).sum(dim=-1)  # (B, T-1)

            valid_mask = (shift_labels != -100).float()
            entropy = (token_entropy * valid_mask).sum() / valid_mask.sum().clamp_min(1.0)

        return loss, entropy

    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, question_lengths = batch

        loss, entropy = self._loss_and_entropy(
            input_ids=input_ids,
            attention_mask=attention_mask,
            question_lengths=question_lengths,
        )

        lr = self.trainer.optimizers[0].param_groups[0]["lr"]        
        
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_entropy", entropy, prog_bar=True)
        self.log("lr", lr, prog_bar=False)

        return loss

    def validation_step(self, batch, batch_idx):
        input_ids, attention_mask, question_lengths = batch

        loss, entropy = self._loss_and_entropy(
            input_ids=input_ids,
            attention_mask=attention_mask,
            question_lengths=question_lengths,
        )

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_entropy", entropy, prog_bar=True)
        
        return {"val_loss": loss.detach(), "val_entropy": entropy.detach()}

    @torch.no_grad()
    def generate_from_prompts(self, prompts: List[str]) -> List[dict]:
        self.eval()

        results = []
        device = self.device

        for prompt in prompts:
            encoded = self.tokenizer(
                prompt,
                return_tensors="pt",
                add_special_tokens=True,
            )
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)

            generated_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.hparams.generation_max_new_tokens,
                do_sample=True,
                temperature=self.hparams.generation_temperature,
                top_k=self.hparams.generation_top_k,
                top_p=self.hparams.generation_top_p,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

            full_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)

            # completion-only text
            prompt_len = input_ids.shape[1]
            completion_ids = generated_ids[0, prompt_len:]
            completion_text = self.tokenizer.decode(completion_ids, skip_special_tokens=True)

            results.append(
                {
                    "prompt": prompt,
                    "completion": completion_text,
                    "full_text": full_text,
                }
            )

        return results


# ============================================================
# Callback: log generations to MLflow whenever validation runs
# ============================================================
class MLflowGenerationCallback(Callback):
    def __init__(
        self,
        prompts: Optional[List[str]] = None,
        artifact_dir: str = "generations",
        wrap_width: int = 80,
    ):
        super().__init__()
        self.prompts = prompts or []
        self.artifact_dir = artifact_dir
        self.wrap_width = wrap_width

    def _wrap_block(self, text: str) -> str:
        text = (text or "").strip()
        if not text:
            return ""

        lines = []
        for para in text.splitlines():
            para = para.strip()
            if not para:
                lines.append("")
            else:
                lines.append(
                    textwrap.fill(
                        para,
                        width=self.wrap_width,
                        break_long_words=False,
                        break_on_hyphens=False,
                    )
                )
        return "\n".join(lines)

    def _format_generations(self, trainer, pl_module, results: List[dict]) -> str:
        parts = [
            f"max_seq_len: {pl_module.hparams.generation_max_new_tokens}",
            f"top_k: {pl_module.hparams.generation_top_k}",
            f"top_p: {pl_module.hparams.generation_top_p}",
            f"temperature: {pl_module.hparams.generation_temperature}",
            "",
            "",
        ]

        for i, item in enumerate(results):
            parts.append("+++++++++++++")
            parts.append("")
            parts.append("prompt:")
            parts.append(self._wrap_block(item["prompt"]))
            parts.append("")
            parts.append("---")
            parts.append("")
            parts.append("completion:")
            parts.append(self._wrap_block(item["completion"]))
            parts.append("")

        return "\n".join(parts).rstrip() + "\n"
    
    def on_validation_epoch_end(self, trainer, pl_module):
        # if trainer.sanity_checking:
        #     return

        if not self.prompts:
            return

        logger = trainer.logger
        if logger is None or not isinstance(logger, MLFlowLogger):
            return

        results = pl_module.generate_from_prompts(self.prompts)
        formatted_text = self._format_generations(trainer, pl_module, results)

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(
                tmpdir,
                f"generations_epoch_{trainer.current_epoch:04d}_step_{trainer.global_step:08d}.txt",
            )
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(formatted_text)

            logger.experiment.log_artifact(
                run_id=logger.run_id,
                local_path=out_path,
                artifact_path=self.artifact_dir,
            )


def main(
    args: argparse.ArgumentParser,
    tokenizer: AutoTokenizer,
):
    
    run_name = args.run_prefix + "-" + pd.Timestamp.now().strftime("%Y-%m-%d-%H%M%S")

    datamodule = SFTDataModule(
        data_dir=DATA_DIR,
        pad_token_id=tokenizer.pad_token_id,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
        
    module = GPT2SFTModule(
        tokenizer,
        model_name=MODEL_NAME,
        lr=args.lr,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        generation_prompts=GENERATION_PROMPTS,
        generation_max_new_tokens=args.gen_max_tokens,
        generation_temperature=args.gen_temperature,
        generation_top_k=args.gen_top_k,
        generation_top_p=args.gen_top_p,
    )
    
    mlf_logger = MLFlowLogger(
        experiment_name=args.exp_name,
        tracking_uri=f"file:{BASE_DIR}/mlruns",  # Colab-local (ephemeral) filesystem
        run_name=run_name,
    )
    
    checkpoint_cb = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        dirpath=f"{BASE_DIR}/mlruns/{mlf_logger.experiment_id}/{mlf_logger.run_id}/artifacts/",
        filename="best-{step}-{val_loss:.4f}",
    )

    generation_callback = MLflowGenerationCallback(
        prompts=GENERATION_PROMPTS,
        artifact_dir="generations",
    )
    
    # ============================================================
    # Trainer
    # ============================================================
    trainer = pl.Trainer(
        max_steps=args.max_steps,
        max_epochs=-1,
        logger=mlf_logger,
        callbacks=[
            checkpoint_cb,
            generation_callback,
            LogBestCkptAndPyfuncToMLflow(module_cls=GPT2SFTModule, register_name=args.model_name),
        ],
        accelerator="auto",
        devices="auto",
        precision="16-mixed" if torch.cuda.is_available() else "32-true",
        log_every_n_steps=args.log_interval,
        val_check_interval=args.val_interval*args.grad_accum_step,   # validation once per epoch
        check_val_every_n_epoch=None,
        enable_checkpointing=True,
        # num_sanity_val_steps=0,
        limit_val_batches=args.val_batches,
        accumulate_grad_batches=args.grad_accum_step,
        gradient_clip_val=args.grad_clip,    
    )

    # ============================================================
    # Log a few static params explicitly
    # ============================================================
    
    mlf_logger.log_hyperparams(
        {
            "model_name": MODEL_NAME,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "lr": args.lr,
            "warmup_ratio": args.warmup_ratio,
            "weight_decay": args.weight_decay,
            "max_steps": args.max_steps,
            "generation_max_new_tokens": args.gen_max_tokens,
            "generation_temperature": args.gen_temperature,
            "generation_top_k": args.gen_top_k,
            "generation_top_p": args.gen_top_p,
            "num_generation_prompts": len(GENERATION_PROMPTS),
            "accumulate_grad_steps": args.grad_accum_step,
            "grad_clip": args.grad_clip,
        }
    )
    
    trainer.fit(module, datamodule=datamodule)

    print("MLflow run_id:", mlf_logger.run_id)
    print("Best checkpoint:", checkpoint_cb.best_model_path)
    print("Best val_loss:", checkpoint_cb.best_model_score.item() 
          if checkpoint_cb.best_model_score is not None else None)
    print("Last checkpoint:", checkpoint_cb.last_model_path)


if __name__ == "__main__":    
    parser = argparse.ArgumentParser(
        description="Perform Supervised Fine-Tuning on GPT2"
    )
    parser.add_argument("--exp-name", type=str, required=True, help="MLFlow experiment name")
    parser.add_argument("--run-prefix", type=str, required=True, help="MLFlow run name prefix")
    parser.add_argument("--model-name", type=str, required=True, help="MLFlow model name")
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS, help="maximum number of steps")
    parser.add_argument("--lr", type=float, default=LR, help="learning rate")
    parser.add_argument("--warmup-ratio", type=float, default=WARMUP_RATIO, help="warmup ratio")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="train batch size")
    parser.add_argument("--grad-accum-step", type=int, default=GRAD_ACC_STEPS, help="gradient accumulation step")
    parser.add_argument("--grad-clip", type=float, default=GRAD_CLIP, help="gradient clipping")
    parser.add_argument("--log-interval", type=int, default=LOG_EVERY_N_STEPS, help="logging interval")
    parser.add_argument("--val-interval", type=int, default=VAL_CHECK_INTERVAL, help="evaluation logging interval")
    parser.add_argument("--val-batches", type=int, default=VAL_BATCHES, help="max number of evaluation batches")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY, help="weighted decay for gradient desc.")
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS, help="Number of dataloader workers")

    parser.add_argument("--gen-max-tokens", type=int, default=GEN_MAX_NEW_TOKENS, help="maximum number of generated tokens")
    parser.add_argument("--gen-temperature", type=float, default=GEN_TEMPERATURE, help="generated sampling temperature")
    parser.add_argument("--gen-top-k", type=int, default=GEN_TOP_K, help="sample top-k of generated tokens")
    parser.add_argument("--gen-top-p", type=float, default=GEN_TOP_P, help="sample top-p of generated tokens")
    
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    main(args, tokenizer)
