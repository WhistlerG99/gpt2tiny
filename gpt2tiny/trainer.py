import os
import torch
from dataclasses import dataclass
from .model import GPT2
import pytorch_lightning as pl


@dataclass
class TrainingConfig:
    learning_rate: float = 6e-4
    max_iters: int = 30000
    weight_decay: float = 1e-1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0

    decay_lr: bool = True
    warmup_iters: int = 1000
    lr_decay_iters: int = 30000
    min_lr: float = 6e-5

    eval_interval: int = 100
    log_interval: int = 10
    eval_iters: int = 200
    gradient_accumulation_steps: int = 4
    batch_size: int = 64
    num_workers: int = 1

    device: str = str(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    dtype: str = "bfloat16"
    compile: bool = True


class GPT2Module(pl.LightningModule):
    def __init__(
        self,
        config,
        tokenizer,
        prompt: str = "A dragon in a cave",
        max_seq_len: int = 128,
        top_k: int = 50,
        top_p: float = 0.95,
        temperature: float = 0.8,
        gen_every_n_epochs: int = 1,
    ):
        super().__init__()
        self.model = GPT2(config)
        self.tokenizer = tokenizer

        self.prompt = prompt
        self.max_seq_len = max_seq_len
        self.top_k = top_k
        self.top_p = top_p
        self.temperature = temperature
        self.gen_every_n_epochs = gen_every_n_epochs

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits, loss = self.model(x, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits, loss = self.model(x, y)
        self.log("val_loss", loss, prog_bar=True)

        if (
            batch_idx == 0
            and self.trainer.is_global_zero
            and (self.current_epoch % self.gen_every_n_epochs == 0)
        ):
            self._log_generation_to_mlflow()

        return loss

    @torch.no_grad()
    def _log_generation_to_mlflow(self):
        self.model.eval()

        text = self.model.generate(
            self.prompt,
            self.max_seq_len,
            top_k=self.top_k,
            top_p=self.top_p,
            tokenizer=self.tokenizer,
            temperature=self.temperature,
        )

        # --- log to MLflow ---
        logger = getattr(self, "logger", None)
        mlf = getattr(logger, "experiment", None) if logger is not None else None
        run_id = getattr(logger, "run_id", None) if logger is not None else None
        if mlf is None or run_id is None:
            # Not using MLFlowLogger (or no active run)
            self.print(f"[epoch={self.current_epoch}] sample generation:\n{text}\n")
            return

        os.makedirs("mlflow_artifacts", exist_ok=True)
        fname = f"gen_epoch_{self.current_epoch:04d}_step_{self.global_step:09d}.txt"
        fpath = os.path.join("mlflow_artifacts", fname)

        with open(fpath, "w", encoding="utf-8") as f:
            f.write(f"prompt: {self.prompt}\n")
            f.write(f"max_seq_len: {self.max_seq_len}\n")
            f.write(f"top_k: {self.top_k}\n")
            f.write(f"top_p: {self.top_p}\n")
            f.write(f"temperature: {self.temperature}\n")
            f.write("\n---\n")
            f.write(text)

        mlf.log_artifact(run_id, fpath, artifact_path="generations")

        # Optional: short preview as a tag
        preview = str(text).replace("\n", " ")[:200]
        mlf.set_tag(run_id, f"gen_preview_epoch_{self.current_epoch}", preview)

    def configure_optimizers(self, lr=0.001):
        return torch.optim.AdamW(self.parameters(), lr=lr)
