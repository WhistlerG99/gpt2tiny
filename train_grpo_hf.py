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

from gpt2tiny.dataset import PromptDataset 
from gpt2tiny.callbacks import (
    LogBestCkptAndPyfuncToMLflow,
    MLflowGenerationCallback,
)
from gpt2tiny.trainer import GPT2GRPOModule

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
from pytorch_lightning.loggers import MLFlowLogger
from transformers import AutoTokenizer

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
class GRPODataModule(pl.LightningDataModule):
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
            PromptDataset(
                split="train",
                data_dir=self.data_dir,
                weights="Balanced",
            ),
            collate_fn=lambda batch: PromptDataset.collator(batch, pad_id=self.pad_token_id),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    def val_dataloader(self):
        return DataLoader(
            PromptDataset(
                split="validation",
                data_dir=self.data_dir,
                weights="Balanced",
            ),
            collate_fn=lambda batch: PromptDataset.collator(batch, pad_id=self.pad_token_id),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
    

def main(
    args: argparse.ArgumentParser,
    tokenizer: AutoTokenizer,
):
    
    run_name = args.run_prefix + "-" + pd.Timestamp.now().strftime("%Y-%m-%d-%H%M%S")

    datamodule = GRPODataModule(
        data_dir=DATA_DIR,
        pad_token_id=tokenizer.pad_token_id,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
        
    module = GPT2GRPOModule(
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
            LogBestCkptAndPyfuncToMLflow(module_cls=GPT2GRPOModule, register_name=args.model_name),
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
        description="Perform GRPO on GPT2"
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
