import os
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List
import mlflow

from gpt2tiny.tokenizer import Tokenizer
from gpt2tiny.model import GPTConfig
from gpt2tiny.dataset import PreTokDataset 
from gpt2tiny.trainer import PreTrainGPT2Module
from gpt2tiny.callbacks import LogBestCkptAndPyfuncToMLflow#, SetCheckpointDirCallback

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
from pytorch_lightning.loggers import MLFlowLogger


BASE_DIR = "/teamspace/studios/this_studio/gpt2tiny"
DATA_CACHE_DIR = Path(BASE_DIR) / "data"

DATA_DIR = [DATA_CACHE_DIR / "TinyStories_all_data_only_pretrain", DATA_CACHE_DIR / "TinyStories_all_data_only_sft"]

BATCH_SIZE = 64
NUM_WORKERS = 4
LR = 3e-5
WARMUP_RATIO=0.05
WEIGHT_DECAY = 0.01
MAX_STEPS = 20000
VAL_BATCHES = 50
VAL_CHECK_INTERVAL=25
LOG_EVERY_N_STEPS=25
GRAD_ACC_STEPS=4
GRAD_CLIP=1.0

BLOCK_SIZE = 512
N_LAYER = 8
N_HEAD = 8
N_EMBED = 512
DROPOUT = 0.2

# Generation settings used whenever validation runs
GEN_MAX_NEW_TOKENS = 128
GEN_TEMPERATURE = 0.8
GEN_TOP_K = 50
GEN_TOP_P = 0.95

GENERATION_PROMPTS = [
    "A dragon in a cave.",
    "A girl and her cat.",
    "The boy went to school.",
    "A bear and a rabbit are friends.",
    "The king went to",
]

# Precision / hardware
MATMUL_PRECISION = "medium"   # "medium" or "high"
torch.set_float32_matmul_precision(MATMUL_PRECISION)


class PreTrainDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: Path|str|List[Path|str],
        block_size: int = 512,
        batch_size: int = 64,
        num_workers: int = 1,
    ):
        super().__init__()
        if isinstance(data_dir, list):
            self.data_dir = [Path(d) for d in data_dir]
        else:
            self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.block_size = block_size

    def train_dataloader(self):
        return DataLoader(
            PreTokDataset(
                self.block_size,
                split="train",
                data_dir=self.data_dir,
                weights="Balanced",
            ),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    def val_dataloader(self):
        return DataLoader(
            PreTokDataset(
                self.block_size,
                split="validation",
                data_dir=self.data_dir,
                weights="Balanced",
            ),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

def main(
    args: argparse.ArgumentParser,
    tokenizer: Tokenizer,
):
    
    run_name = args.run_prefix + "-" + pd.Timestamp.now().strftime("%Y-%m-%d-%H%M%S")

    datamodule = PreTrainDataModule(
        data_dir=DATA_DIR,
        block_size=args.block_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    model_config = GPTConfig(
        block_size=args.block_size,
        vocab_size=tokenizer.n_words,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embed=args.n_embed,
        dropout=args.dropout,
    )

    module = PreTrainGPT2Module(
        model_config,
        tokenizer,
        gen_every_n_epochs=500,
        prompts=GENERATION_PROMPTS,
        max_seq_len=args.gen_max_tokens,
        temperature=args.gen_temperature,
        top_k=args.gen_top_k,
        top_p=args.gen_top_p,
        lr=args.lr,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
    )

    module = torch.compile(module).train()
    
    mlf_logger = MLFlowLogger(
        experiment_name=args.exp_name,
        tracking_uri=f"file:{BASE_DIR}/mlruns",  # Colab-local (ephemeral) filesystem
        run_name=run_name,
    )
    
    # crucial change: set dirpath to mlf_logger.log_dir
    # this makes ModelCheckpoint save files into the local directory that MLFlowLogger monitors for artifacts
    checkpoint_cb = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        dirpath=f"{BASE_DIR}/mlruns/{mlf_logger.experiment_id}/{mlf_logger.run_id}/artifacts/",
        filename="best-{step}-{val_loss:.4f}",
    )
    
    
    trainer = pl.Trainer(
        max_steps=args.max_steps,
        max_epochs=-1,
        logger=mlf_logger,
        callbacks=[
            checkpoint_cb,
            LogBestCkptAndPyfuncToMLflow(module_cls=PreTrainGPT2Module, register_name=args.model_name),
        ],        
        accelerator="auto",
        devices="auto",
        precision="16-mixed" if torch.cuda.is_available() else "32-true",
        log_every_n_steps=args.log_interval,
        val_check_interval=args.val_interval*args.grad_accum_step,   # validation once per epoch
        check_val_every_n_epoch=None,
        enable_checkpointing=True,
        limit_val_batches=args.val_batches,
        accumulate_grad_batches=args.grad_accum_step,
        gradient_clip_val=args.grad_clip,

    )

    # ============================================================
    # Log a few static params explicitly
    # ============================================================
    
    mlf_logger.log_hyperparams(
        {
            "model_name": args.model_name,
            "block_size": args.block_size,
            "vocab_size": tokenizer.n_words,
            "tokenizer_path": args.tokenizer_path,
            "n_layer": args.n_layer,
            "n_head": args.n_head,
            "n_embed": args.n_embed,
            "dropout": args.dropout,
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
        description="Perform pretraining on a GPT2 type model"
    )    

    parser.add_argument("--exp-name", type=str, required=True, help="MLFlow experiment name")
    parser.add_argument("--run-prefix", type=str, required=True, help="MLFlow run name prefix")
    parser.add_argument("--model-name", type=str, required=True, help="MLFlow model name")
    parser.add_argument("--tokenizer-path", type=str, required=True, help="Path to the tokenizer model")

    parser.add_argument("--block-size", type=int, default=BLOCK_SIZE, help="Max number of tokens in each sequence")
    parser.add_argument("--n-layer", type=int, default=N_LAYER, help="Number of transformer layers")
    parser.add_argument("--n-head", type=int, default=N_HEAD, help="Number of attention heads")
    parser.add_argument("--n-embed", type=int, default=N_EMBED, help="Embedding dimension")
    parser.add_argument("--dropout", type=float, default=DROPOUT, help="Dropout probability")

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

    tokenizer = Tokenizer(args.tokenizer_path)
    # tokenizer = Tokenizer(f"{BASE_DIR}/data/tok4096_tinystories.model")

    main(args, tokenizer)
