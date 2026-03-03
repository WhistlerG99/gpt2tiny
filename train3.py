# train.py
import os
import shutil
from pathlib import Path
from typing import Optional, Literal

import numpy as np
import pandas as pd
import mlflow

from gpt2tiny.tokenizer import Tokenizer
from gpt2tiny.model import GPT2, GPTConfig
from gpt2tiny.dataset import PreTokDataset
from gpt2tiny.trainer import GPT2Module, TrainingConfig
from gpt2tiny.callbacks import UploadLastAndBestToMLflow

from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
from pytorch_lightning.loggers import MLFlowLogger


BASE_DIR = "/teamspace/studios/this_studio/gpt2tiny"
DATA_CACHE_DIR = Path(BASE_DIR) / "data"

LOCAL_RUN_DIR = Path(BASE_DIR) / "lightning_runs"  # or Path("/tmp/lightning_runs")
LOCAL_RUN_DIR.mkdir(parents=True, exist_ok=True)

MLFLOW_TRACKING_URI = "https://5000-01kgr6z0qq5h0srek4vj1jq4pb.cloudspaces.litng.ai"

os.environ["MLFLOW_ARTIFACT_URI"] = "file:///teamspace/s3_folders/mlflow-job-artifacts"
os.environ["MLFLOW_TRACKING_URI"] = MLFLOW_TRACKING_URI
os.environ["MLFLOW_ENABLE_PROXY_MULTIPART_UPLOAD"] = "true"


def _build_dataloaders(model_config: GPTConfig, trainer_config: TrainingConfig):
    train_dataloader = DataLoader(
        PreTokDataset(
            model_config.block_size,
            split="train",
            data_dir=[DATA_CACHE_DIR / "TinyStories_all_data_only_pretrain"],
            weights="Balanced",
        ),
        batch_size=trainer_config.batch_size,
        num_workers=trainer_config.num_workers,
    )

    eval_dataloader = DataLoader(
        PreTokDataset(
            model_config.block_size,
            split="validation",
            data_dir=[DATA_CACHE_DIR / "TinyStories_all_data_only_pretrain"],
            weights="Balanced",
        ),
        batch_size=trainer_config.batch_size,
        num_workers=trainer_config.num_workers,
    )

    return train_dataloader, eval_dataloader





def main(
    experiment_name: str,
    run_name: str,
    model_name: str,
    tokenizer: Tokenizer,
    model_config: GPTConfig,
    trainer_config: TrainingConfig,
    resume_from_run_name: Optional[str] = None,
    resume_which: Literal["last", "best"] = "last",
):
    train_dataloader, eval_dataloader = _build_dataloaders(model_config, trainer_config)

    model = GPT2Module(
        model_config,
        tokenizer,
        gen_every_n_epochs=500,
        prompts=[
            "A dragon in a cave",
            "1+1 is",
            "what is the gcd of 21 and 36?",
        ],
    )

    # IMPORTANT: stop Lightning from auto-uploading every checkpoint (S3 pile-up)
    mlf_logger = MLFlowLogger(
        experiment_name=experiment_name,
        tracking_uri=MLFLOW_TRACKING_URI,
        run_name=run_name,
        log_model=False,
    )

    ckpt_dir = LOCAL_RUN_DIR / experiment_name / run_name / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)    
    # ckpt_dir = Path(mlf_logger.log_dir) / "checkpoints"
    # ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Keep only one "best" checkpoint locally + always write last.ckpt locally
    checkpoint_cb = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=True,
        dirpath=str(ckpt_dir),
        filename="best-{step}-{val_loss:.4f}",
    )

    resume_ckpt_path = None
    resume_dst_dir = LOCAL_RUN_DIR / experiment_name / run_name / "resume_ckpts"
    if resume_from_run_name:
        # Download last/best from the *previous* run name and resume from it.
        resume_ckpt_path = _download_resume_ckpt(
            experiment_name=experiment_name,
            run_name=resume_from_run_name,
            which=resume_which,
            dst_dir=resume_dst_dir,
            tracking_uri=MLFLOW_TRACKING_URI,
        )
        print(f"Resuming from run_name='{resume_from_run_name}' ({resume_which}) -> {resume_ckpt_path}")

    trainer = pl.Trainer(
        accelerator="auto",
        devices=1,
        precision="16-mixed",
        max_steps=trainer_config.max_iters,
        val_check_interval=trainer_config.eval_interval,
        limit_val_batches=200,
        logger=mlf_logger,
        callbacks=[
            checkpoint_cb,
            UploadLastAndBestToMLflow(artifact_subdir="checkpoints", upload_every_n_val=1),
            # Keep your existing "log pyfunc + register" callback if you want.
            # If it logs a separate checkpoint artifact, ensure it also uses fixed names, or only logs at end.
            # LogBestCkptAndPyfuncToMLflow(module_cls=GPT2Module, register_name=model_name),
        ],
        log_every_n_steps=trainer_config.log_interval,
        accumulate_grad_batches=trainer_config.gradient_accumulation_steps,
        gradient_clip_val=trainer_config.grad_clip,
    )

    print("MLflow tracking uri (global):", mlflow.get_tracking_uri())

    trainer.fit(model, train_dataloader, eval_dataloader, ckpt_path=str(resume_ckpt_path) if resume_ckpt_path else None)


if __name__ == "__main__":
    model_config = GPTConfig(flash=True, block_size=64)
    trainer_config = TrainingConfig(
        batch_size=64,
        num_workers=4,
        max_iters=5000,
        log_interval = 100,
        eval_interval = 500,
    )

    experiment_name = "test"

    # New run name for this attempt
    run_name = "tinystories-pretrain"
    run_name += "-" + pd.Timestamp.now().strftime("%Y-%m-%d-%H%M%S")

    tokenizer = Tokenizer(f"{BASE_DIR}/data/tok4096_tinystories.model")
    model_name = "GPT2Pretrained"

    # ---- Resume controls (edit or wire to argparse) ----
    RESUME_FROM_RUN_NAME = None  # e.g. "tinystories-pretrain-2026-02-26-221500"
    RESUME_WHICH = "last"        # "last" or "best"
    # -----------------------------------------------

    main(
        experiment_name=experiment_name,
        run_name=run_name,
        model_name=model_name,
        tokenizer=tokenizer,
        model_config=model_config,
        trainer_config=trainer_config,
        resume_from_run_name=RESUME_FROM_RUN_NAME,
        resume_which=RESUME_WHICH,
    )