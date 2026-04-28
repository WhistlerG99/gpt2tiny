import os
import shutil
import textwrap
import tempfile
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, List
import mlflow
import mlflow.pyfunc
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import MLFlowLogger
from importlib.metadata import version

pkg_version = version("gpt2tiny")

class UploadLastAndBestToMLflow(Callback):
    """
    Uploads (overwriting keys each time, so no pile-up):
      - checkpoints/last.ckpt  (updated on each validation end)
      - checkpoints/best.ckpt  (updated only when best changes)
    """

    def __init__(self, artifact_subdir: str = "checkpoints", upload_every_n_val: int = 1):
        super().__init__()
        self.artifact_subdir = artifact_subdir
        self.upload_every_n_val = upload_every_n_val
        self._val_calls = 0
        self._last_best_path: Optional[str] = None

    def _ensure_mlflow_context(self, trainer: pl.Trainer) -> str:
        logger = trainer.logger
        if not isinstance(logger, MLFlowLogger):
            raise RuntimeError("UploadLastAndBestToMLflow requires MLFlowLogger.")

        # Compatibility across Lightning versions
        tracking_uri = getattr(logger, "tracking_uri", None) or getattr(logger, "_tracking_uri", None)
        if not tracking_uri:
            raise RuntimeError("Could not determine MLflow tracking URI from MLFlowLogger.")

        mlflow.set_tracking_uri(tracking_uri)

        # experiment_name may also differ
        exp_name = getattr(logger, "experiment_name", None) or getattr(logger, "_experiment_name", None)
        if exp_name:
            mlflow.set_experiment(exp_name)

        # run_id is stable on MLFlowLogger
        run_id = getattr(logger, "run_id", None)
        if not run_id:
            raise RuntimeError("Could not determine run_id from MLFlowLogger.")
        return run_id

    @staticmethod
    def _log_as_fixed_name(src_path: str, fixed_name: str, artifact_path: str) -> None:
        src = Path(src_path)
        if not src.exists():
            return
    
        dst = src.parent / fixed_name
    
        # If src already has the fixed name (e.g., last.ckpt), don't copy onto itself
        if src.resolve() == dst.resolve():
            mlflow.log_artifact(str(src), artifact_path=artifact_path)
            return
    
        shutil.copy2(src, dst)  # fixed filename => stable artifact key (overwrites)
        try:
            mlflow.log_artifact(str(dst), artifact_path=artifact_path)
        finally:
            dst.unlink(missing_ok=True)

    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if hasattr(trainer, "is_global_zero") and not trainer.is_global_zero:
            return

        self._val_calls += 1
        if self._val_calls % self.upload_every_n_val != 0:
            return

        run_id = self._ensure_mlflow_context(trainer)
        ckpt_cb = trainer.checkpoint_callback

        last_path = getattr(ckpt_cb, "last_model_path", None) if ckpt_cb else None
        best_path = getattr(ckpt_cb, "best_model_path", None) if ckpt_cb else None

        with mlflow.start_run(run_id=run_id):
            if last_path:
                self._log_as_fixed_name(last_path, "last.ckpt", self.artifact_subdir)

            if best_path and best_path != self._last_best_path:
                self._log_as_fixed_name(best_path, "best.ckpt", self.artifact_subdir)
                self._last_best_path = best_path


class SetCheckpointDirCallback(Callback):
    def __init__(self, checkpoint_callback_instance, subdirectory="checkpoints"):
        self.checkpoint_callback_instance = checkpoint_callback_instance
        self.subdirectory = subdirectory

    def on_train_start(self, trainer, pl_module):
        print(f"[{self.__class__.__name__}] -- on_train_start called.")
        if isinstance(trainer.logger, MLFlowLogger):
            mlf_logger = trainer.logger
            # mlflow.active_run() should definitely be available here
            if mlflow.active_run() is not None:
                mlflow_log_dir = mlf_logger.log_dir
                print(f"[{self.__class__.__name__}] -- MLFlowLogger log_dir: {mlflow_log_dir}")

                if mlflow_log_dir:
                    new_dirpath = os.path.join(mlflow_log_dir, self.subdirectory)
                    os.makedirs(new_dirpath, exist_ok=True)
                    self.checkpoint_callback_instance.dirpath = new_dirpath
                    print(f"[{self.__class__.__name__}] -- ModelCheckpoint dirpath set to: {self.checkpoint_callback_instance.dirpath}")
                else:
                    print(f"[{self.__class__.__name__}] -- warning: mlflow_log_dir is none, ModelCheckpoint will use default dirpath or fall back.")
            else:
                print(f"[{self.__class__.__name__}] -- MLFlowLogger active run is not initialized. This should not happen in on_train_start.")
        else:
            print(f"[{self.__class__.__name__}] -- warning: logger is not an MLFlowLogger instance.")

# class SetCheckpointDirCallback(Callback):
#     def __init__(self, checkpoint_callback_instance, subdirectory="checkpoints"):
#         self.checkpoint_callback_instance = checkpoint_callback_instance
#         self.subdirectory = subdirectory

#     def on_setup(self, trainer, pl_module, stage: str):
#         # ensure the logger is an MLFlowLogger and that it has an experiment set up
#         if isinstance(trainer.logger, MLFlowLogger) and trainer.logger.experiment is not None:
#             # mlf_logger.log_dir is the local temporary directory where artifacts are staged
#             # before being uploaded to the remote MLflow artifact store.
#             # this ensures checkpoints are saved into a location MLflow is monitoring.
#             mlflow_log_dir = trainer.logger.log_dir
#             if mlflow_log_dir:
#                 new_dirpath = os.path.join(mlflow_log_dir, self.subdirectory)
#                 os.makedirs(new_dirpath, exist_ok=True)
#                 self.checkpoint_callback_instance.dirpath = new_dirpath
#             else:
#                 print("warning: mlflow_log_dir is none, ModelCheckpoint will use default dirpath or fall back.")
#         else:
#             print("warning: mlf_logger not found or not initialized. ModelCheckpoint dirpath might not be set correctly for mlflow artifacts.")


class LightningPyfunc(mlflow.pyfunc.PythonModel):
    def __init__(self, module_cls):
        self.module_cls = module_cls
        self.model = None

    def load_context(self, context):
        self.model = self.module_cls.load_from_checkpoint(context.artifacts["checkpoint"], weights_only=False)
        self.model.eval()

    def predict(self, context, model_input):
        if isinstance(model_input, pd.DataFrame):
            x = torch.tensor(model_input.to_numpy(), dtype=torch.float32)
        else:
            x = torch.tensor(np.asarray(model_input), dtype=torch.float32)
        with torch.no_grad():
            y = self.model(x)
        return y.detach().cpu().numpy() if torch.is_tensor(y) else y


class LogBestCkptAndPyfuncToMLflow(Callback):
    def __init__(
        self,
        module_cls,
        ckpt_artifact_subdir="checkpoints",
        pyfunc_artifact_path="model",
        register_name: str | None = None,   # set to "MyModel" if you want auto-register
    ):
        self.module_cls = module_cls
        self.ckpt_artifact_subdir = ckpt_artifact_subdir
        self.pyfunc_artifact_path = pyfunc_artifact_path
        self.register_name = register_name

    def on_fit_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        if hasattr(trainer, "is_global_zero") and not trainer.is_global_zero:
            return

        ckpt_cb = trainer.checkpoint_callback
        best_ckpt = ckpt_cb.best_model_path if ckpt_cb else ""
        if not best_ckpt:
            raise RuntimeError("No best checkpoint found. Ensure ModelCheckpoint(save_top_k=1, monitor=...) is set.")

        run_id = trainer.logger.run_id  # MLFlowLogger exposes this
        
        with mlflow.start_run(run_id=run_id):

            mlflow.log_artifact(best_ckpt, artifact_path=self.ckpt_artifact_subdir)

            mlflow.pyfunc.log_model(
                artifact_path=self.pyfunc_artifact_path,
                python_model=LightningPyfunc(self.module_cls),
                artifacts={"checkpoint": best_ckpt},
                code_paths=[str(Path(__import__("gpt2tiny").__file__).resolve().parent)],
                pip_requirements=[
                    "mlflow",
                    "torch",
                    "pytorch-lightning",
                    "pandas",
                    "numpy",
                    f"gpt2tiny=={pkg_version}",
                ],
            )

            if self.register_name:
                model_uri = f"runs:/{run_id}/{self.pyfunc_artifact_path}"
                mlflow.register_model(model_uri=model_uri, name=self.register_name)


# ============================================================
# Callback: log generations from Huggingface GPT2 Model to MLflow whenever validation runs
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
            if "reward" in item:
                parts.append("")
                parts.append("reward:")
                parts.append(item["reward"])
                parts.append("")
                parts.append("---")
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
    
    def _generate_and_log_completions(self, trainer, pl_module):
        # if trainer.sanity_checking:
        #     return

        if not self.prompts:
            return

        logger = trainer.logger
        if logger is None or not isinstance(logger, MLFlowLogger):
            return

        # Ensure correct mode + no grad
        was_training = pl_module.training
        pl_module.eval()
        
        results = pl_module.generate_from_prompts(self.prompts)

        if was_training:
            pl_module.train()        
        
        formatted_text = self._format_generations(trainer, pl_module, results)

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(
                tmpdir,
                f"generations_step_{trainer.global_step:08d}.txt",
            )
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(formatted_text)

            logger.experiment.log_artifact(
                run_id=logger.run_id,
                local_path=out_path,
                artifact_path=self.artifact_dir,
            )

    def on_validation_epoch_end(self, trainer, pl_module):
        self._generate_and_log_completions(trainer, pl_module)

    # def on_train_start(self, trainer, pl_module):
    #     datamodule = trainer.datamodule
    #     if datamodule is None:
    #         return
    
    #     val_loader = datamodule.val_dataloader()
    
    #     was_training = pl_module.training
    #     pl_module.eval()
    
    #     with torch.no_grad():
    #         for batch_idx, batch in enumerate(val_loader):
    #             batch = trainer.strategy.batch_to_device(batch, pl_module.device)
    #             pl_module.validation_step(batch, batch_idx)
    
    #     # run your generation logging once
    #     self._generate_and_log_completions(trainer, pl_module)
    
    #     if was_training:
    #         pl_module.train()
