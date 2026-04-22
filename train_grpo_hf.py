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
from gpt2tiny.rewards.embedding import (
    RewardWeights,
    StoryRewardConfig,
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
DATA_DIR = DATA_CACHE_DIR / "TinyStories_prompts_huggingface_gpt2_v2"

BATCH_SIZE = 2
NUM_WORKERS = 4

NUM_GEN = 8
MAX_SEQ_LEN = 248
TEMPERATURE = 1.0
TOP_K = 50
TOP_P = 0.95

LR = 3e-5
WARMUP_RATIO=0.05
WEIGHT_DECAY = 0.01
MAX_STEPS = 1000
VAL_BATCHES = 16
VAL_CHECK_INTERVAL=5
LOG_EVERY_N_STEPS=25
GRAD_ACC_STEPS=16
GRAD_CLIP=1.0

KL_BETA=0.02
CLIP_EPS=0.2

RW_WORDS = 0.25
RW_POS = 0.15
RW_SUBJECT = 0.20
RW_FEATURES = 0.25
RW_FORMAT = 0.15
RW_COHERENCE = 0.15
RW_PROMPT_COPY_PENALTY = 0.10
RW_META_PENALTY = 0.10
RW_REPETITION_PENALTY = 0.10
RW_STUFFING_PENALTY = 0.10
RW_GIBBERISH_PENALTY = 0.20

RW_MIN_CHARS=80
RW_MAX_CHARS=2000
RW_MIN_SENTENCES=3
RW_MAX_SENTENCES=5

CHECKPOINT_DIR = "./checkpoints"
MLRUNS_DIR = "./mlruns"

# Supply prompts here later
GENERATION_PROMPTS = [   
    {   
        'prompt': "Write a short story. The narrative must use 'handle' in its "
                  "role as a noun, include 'map' as a noun, and use the "
                  "adjective 'loud'.",
        'words': [   
            {'word': 'handle', 'pos': 'noun'},
            {'word': 'map', 'pos': 'noun'},
            {'word': 'loud', 'pos': 'adjective'}],
        'features': [],
        'subject': {},
        'feature_phrases': [],
        'word_clause': "use 'handle' in its role as a noun, include 'map' as a "
                       "noun, and use the adjective 'loud'",
        'feature_clause': None,
        'subject_clause': None
    },
    {   
        'prompt': 'Write a short story. Write about a smuggler who delivers '
                  'food during a blackout in a military outpost. The story '
                  "should use 'prince' in its role as a noun. Also ensure that "
                  'there should be a surprising twist in the plot.',
        'words': [
            {'word': 'prince', 'pos': 'noun'}
        ],
        'features': ['Twist'],
        'subject': {   
            'character': 'smuggler',
            'action': 'delivers food during a blackout',
            'place': 'a military outpost',
            'adjective': None,
            'goal': None
        },
        'feature_phrases': ['there should be a surprising twist in the plot'],
        'word_clause': "use 'prince' in its role as a noun",
        'feature_clause': 'there should be a surprising twist in the plot',
        'subject_clause': 'a smuggler who delivers food during a blackout in a '
                          'military outpost.'
    },
    {   
        'prompt': "Write a short story. The narrative must ensure that 'doll' "
                  'is used as a noun. The story should also satisfy the '
                  'following: the plot should contain a meaningful conflict, '
                  'the narrative should contain a clear moral or takeaway, and '
                  'include setup and payoff somewhere in the story.',
        'words': [{'word': 'doll', 'pos': 'noun'}],
        'features': ['Conflict', 'MoralValue', 'Foreshadowing'],
        'subject': {},
        'feature_phrases': [   
            'the plot should contain a meaningful conflict',
            'the narrative should contain a clear moral or '
            'takeaway',
            'include setup and payoff somewhere in the '
            'story'
        ],
        "word_clause": "ensure that 'doll' is used as a noun",
        "feature_clause": 'the plot should contain a meaningful conflict, the '
                          'narrative should contain a clear moral or takeaway, '
                          'and include setup and payoff somewhere in the story',
        "subject_clause": None
    },
    {
        "prompt": "Compose a narrative. Center it on a wealthy chef who joins a "
                  "circus in an airport. Make sure the narrative use 'believe' in its "
                  "role as a verb and have 'pass' function as a noun. In addition, "
                  "include at least one moment of direct conversation between "
                  "characters, the ending should be unhappy, and there should be a "
                  "clear tension or struggle in the story.",
        "words": [
            {"word": "believe", "pos": "verb"},
            {"word": "pass", "pos": "noun"}
        ],
        "features": ["Dialogue", "BadEnding", "Conflict"],
        "subject": {
            'character': 'chef',
            'action': 'joins a circus',
            'place': 'an airport',
            'adjective': 'wealthy',
            'goal': None
        },
        "feature_phrases": ["include at least one moment of direct conversation "
                            "between characters",
                            "the ending should be unhappy",
                            "there should be a clear tension or struggle in the "
                            "story"],
        "word_clause": "use 'believe' in its role as a verb and have 'pass' function "
                       "as a noun",
        "feature_clause": "include at least one moment of direct conversation between "
                          "characters, the ending should be unhappy, and there should "
                          "be a clear tension or struggle in the story",
        "subject_clause": "a wealthy chef who joins a circus in an airport."
    },
    {
        "prompt": "Create a short fictional story. Additional requirements: include "
                  "at least one moment of direct conversation between characters.",
        "words": [],
        "features": ["Dialogue"],
        "subject": {},
        "feature_phrases": ["include at least one moment of direct conversation "
                            "between characters"],
        "word_clause": None,
        "feature_clause": "include at least one moment of direct conversation between "
                          "characters",
        "subject_clause": None
    },
]

# Generation settings used whenever validation runs
GEN_MAX_NEW_TOKENS = 248
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
        
    # module = GPT2GRPOModule(
    #     tokenizer,
    #     model_name=MODEL_NAME,
    #     num_gen = args.num_gen,
    #     max_seq_len = args.max_seq_len,
    #     temperature = args.temperature,
    #     top_k = args.top_k,
    #     top_p = args.top_p,        
    #     lr=args.lr,
    #     warmup_ratio=args.warmup_ratio,
    #     weight_decay=args.weight_decay,
    #     generation_prompts=GENERATION_PROMPTS,
    #     generation_max_new_tokens=args.gen_max_tokens,
    #     generation_temperature=args.gen_temperature,
    #     generation_top_k=args.gen_top_k,
    #     generation_top_p=args.gen_top_p,
    # )

    reward_weights=RewardWeights(
        words=args.rw_words,
        pos=args.rw_pos,
        subject=args.rw_subject,
        features=args.rw_features,
        format=args.rw_format,
        coherence=args.rw_coherence,
        prompt_copy_penalty=args.rw_prompt_copy_penalty,
        meta_penalty=args.rw_meta_penalty,
        repetition_penalty=args.rw_repetition_penalty,
        stuffing_penalty=args.rw_stuffing_penalty,
        gibberish_penalty=args.rw_gibberish_penalty,
    )

    reward_config = StoryRewardConfig(
        min_chars=args.rw_min_chars,
        max_chars=args.rw_max_chars,
        min_sentences=args.rw_min_sentences,
        max_sentences=args.rw_max_sentences,
    )
    
    module = GPT2GRPOModule.load_from_checkpoint(
        args.init_model_path,
        weights_only=False,
        num_gen = args.num_gen,
        max_seq_len = args.max_seq_len,
        temperature = args.temperature,
        top_k = args.top_k,
        top_p = args.top_p,        
        lr=args.lr,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        generation_prompts=GENERATION_PROMPTS,
        generation_max_new_tokens=args.gen_max_tokens,
        generation_temperature=args.gen_temperature,
        generation_top_k=args.gen_top_k,
        generation_top_p=args.gen_top_p,
        kl_beta=args.kl_beta,
        clip_eps=args.clip_eps,
        reward_weights=reward_weights,
        reward_config=reward_config,
    )

    module = torch.compile(module).train()
    
    mlf_logger = MLFlowLogger(
        experiment_name=args.exp_name,
        tracking_uri=f"file:{BASE_DIR}/mlruns",  # Colab-local (ephemeral) filesystem
        run_name=run_name,
    )
    
    checkpoint_cb = ModelCheckpoint(
        monitor="val_avg_reward",
        mode="max",
        save_top_k=1,
        dirpath=f"{BASE_DIR}/mlruns/{mlf_logger.experiment_id}/{mlf_logger.run_id}/artifacts/",
        filename="best-{step}-{val_avg_reward:.4f}",
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
            "init_model_path": args.init_model_path,
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
            "num_gen": args.num_gen,
            "max_seq_len": args.max_seq_len,
            "temperature": args.temperature,
            "top_k": args.top_k,
            "top_p": args.top_p,
            "kl_beta": args.kl_beta,
            "clip_eps": args.clip_eps,
            "rw_words": args.rw_words,
            "rw_pos": args.rw_pos,
            "rw_subject": args.rw_subject,
            "rw_features": args.rw_features,
            "rw_format": args.rw_format,
            "rw_coherence": args.rw_coherence,
            "rw_prompt_copy_penalty": args.rw_prompt_copy_penalty,
            "rw_meta_penalty": args.rw_meta_penalty,
            "rw_repetition_penalty": args.rw_repetition_penalty,
            "rw_stuffing_penalty": args.rw_stuffing_penalty,
            "rw_gibberish_penalty": args.rw_gibberish_penalty,
            "rw_min_chars": args.rw_min_chars,
            "rw_max_chars": args.rw_max_chars,
            "rw_min_sentences": args.rw_min_sentences,
            "rw_max_sentences": args.rw_max_sentences,
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
        
    parser.add_argument("--init-model-path", type=str, required=True, help="starting model")
    parser.add_argument("--exp-name", type=str, required=True, help="MLFlow experiment name")
    parser.add_argument("--run-prefix", type=str, required=True, help="MLFlow run name prefix")
    parser.add_argument("--model-name", type=str, required=True, help="MLFlow model name")

    parser.add_argument("--max-steps", type=int, default=MAX_STEPS, help="maximum number of steps")

    parser.add_argument("--num-gen", type=int, default=NUM_GEN, help="number of completions to generate for each prompt in GRPO sampling")
    parser.add_argument("--max-seq-len", type=int, default=MAX_SEQ_LEN, help="maximum number of tokens generated in GRPO sampling")
    parser.add_argument("--temperature", type=float, default=TEMPERATURE, help="GRPO sampling temperature")
    parser.add_argument("--top-k", type=int, default=TOP_K, help="top-k for GRPO sampling")
    parser.add_argument("--top-p", type=float, default=TOP_P, help="top-p for GRPO sampling")

    parser.add_argument("--kl-beta", type=float, default=KL_BETA, help="top-p for GRPO sampling")
    parser.add_argument("--clip-eps", type=float, default=CLIP_EPS, help="top-p for GRPO sampling")
    
    
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

    parser.add_argument("--rw-words", type=float, default=RW_WORDS, help="Reward weights (embedding) - required words")
    parser.add_argument("--rw-pos", type=float, default=RW_POS, help="Reward weights (embedding) - required words parts of speech")

    parser.add_argument("--rw-subject", type=float, default=RW_SUBJECT, help="Reward weights (embedding) - subject adherence")
    parser.add_argument("--rw-features", type=float, default=RW_FEATURES, help="Reward weights (embedding) - required features")
    parser.add_argument("--rw-format", type=float, default=RW_FORMAT, help="Reward weights (embedding) - formatting")
    parser.add_argument("--rw-coherence", type=float, default=RW_COHERENCE, help="Reward weights (embedding) - sentence coherence")
    parser.add_argument("--rw-prompt-copy-penalty", type=float, default=RW_PROMPT_COPY_PENALTY, help="Reward weights (embedding) - prompt copy penalty")
    parser.add_argument("--rw-meta-penalty", type=float, default=RW_META_PENALTY, help="Reward weights (embedding) - meta-writing penalty")
    parser.add_argument("--rw-repetition-penalty", type=float, default=RW_REPETITION_PENALTY, help="Reward weights (embedding) - repetition penalty")
    parser.add_argument("--rw-stuffing-penalty", type=float, default=RW_STUFFING_PENALTY, help="Reward weights (embedding) - required word stuffing penalty")
    parser.add_argument("--rw-gibberish-penalty", type=float, default=RW_GIBBERISH_PENALTY, help="Reward weights (embedding) - gibberish penalty")
    
    parser.add_argument("--rw-min-chars", type=int, default=RW_MIN_CHARS, help="Reward config (embedding) - minimum characters")
    parser.add_argument("--rw-max-chars", type=int, default=RW_MAX_CHARS, help="Reward config (embedding) - maximum characters")
    parser.add_argument("--rw-min-sentences", type=int, default=RW_MIN_SENTENCES, help="Reward config (embedding) - minimum sentences")
    parser.add_argument("--rw-max-sentences", type=int, default=RW_MAX_SENTENCES, help="Reward config (embedding) - maximum sentences")
    
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    main(args, tokenizer)
