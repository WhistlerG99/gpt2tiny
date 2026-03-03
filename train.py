import os

import numpy as np
import pandas as pd
from pathlib import Path
import mlflow

from gpt2tiny.tokenizer import Tokenizer
from gpt2tiny.model import GPT2, GPTConfig
from gpt2tiny.dataset import PreTokDataset 
from gpt2tiny.trainer import GPT2Module, TrainingConfig
from gpt2tiny.callbacks import LogBestCkptAndPyfuncToMLflow#, SetCheckpointDirCallback

from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
from pytorch_lightning.loggers import MLFlowLogger


BASE_DIR = "/teamspace/studios/this_studio/gpt2tiny"
DATA_CACHE_DIR = Path(BASE_DIR) / "data"

mlflow_tracking_uri = "https://5000-01kgr6z0qq5h0srek4vj1jq4pb.cloudspaces.litng.ai"
os.environ["MLFLOW_ARTIFACT_URI"] = "file:///teamspace/s3_folders/mlflow-job-artifacts"
os.environ["MLFLOW_TRACKING_URI"] = mlflow_tracking_uri

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
):
    
    # ensure the current run is active for mlflow context
    # this will automatically pick up the artifact_uri set by the MLFlowLogger
    # with mlflow.start_run(run_id=mlf_logger.run_id):
    train_dataloader, eval_dataloader = _build_dataloaders(model_config, trainer_config)
    
    model = GPT2Module(
        model_config,
        tokenizer,
        gen_every_n_epochs=500,
        prompts=[
            "A dragon in a cave",
            "1+1 is",
            "what is the gcd of 21 and 36?"
        ]
    )

    mlf_logger = MLFlowLogger(
        experiment_name=experiment_name,
        tracking_uri=mlflow_tracking_uri,
        # run_name=run_name,
        log_model='all', # this is key! it tells MLFlowLogger to log checkpoints as artifacts
        # artifact_location=f"{mlflow_tracking_uri}/artifacts", # you can explicitly set artifact_location if needed
    )
    
    # crucial change: set dirpath to mlf_logger.log_dir
    # this makes ModelCheckpoint save files into the local directory that MLFlowLogger monitors for artifacts
    checkpoint_cb = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        # dirpath="./local_checkpoints", # example local path
        # dirpath=None,#os.path.join(mlf_logger.log_dir, "checkpoints"), # save within mlflow's local logging dir
        filename="best-{step}-{val_loss:.4f}",
    )
    
    
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
            # LogBestCkptAndPyfuncToMLflow(module_cls=GPT2Module, register_name=model_name),
        #     # SetCheckpointDirCallback(checkpoint_cb) # add our new callback here
        ],
        log_every_n_steps=trainer_config.log_interval,
        accumulate_grad_batches=trainer_config.gradient_accumulation_steps,
        gradient_clip_val=trainer_config.grad_clip,
    )
    
    # print("tracking uri: ", mlflow.get_tracking_uri())
    # print("artifact uri: ", mlflow.get_artifact_uri())
    
    trainer.fit(model, train_dataloader, eval_dataloader)


if __name__ == "__main__":
    model_config = GPTConfig(flash=True, block_size=64)
    trainer_config = TrainingConfig(batch_size=64, num_workers=4, max_iters=500)

    experiment_name = "test"
    run_name = "tinystories-pretrain"
    run_name += "-" + pd.Timestamp.now().strftime("%Y-%m-%d-%H%M%S")

    tokenizer = Tokenizer(f"{BASE_DIR}/data/tok4096_tinystories.model")
    model_name = "GPT2Pretrained"
    main(experiment_name, run_name, model_name, tokenizer, model_config, trainer_config)

    
# import os
# import numpy as np
# import pandas as pd
# from pathlib import Path
# import mlflow

# from gpt2tiny.tokenizer import Tokenizer
# from gpt2tiny.model import GPT2, GPTConfig
# from gpt2tiny.dataset import PreTokDataset
# from gpt2tiny.trainer import GPT2Module, TrainingConfig
# from gpt2tiny.callbacks import LogBestCkptAndPyfuncToMLflow, SetCheckpointDirCallback # import the new callback

# from torch.utils.data import DataLoader
# import pytorch_lightning as pl
# from pytorch_lightning.callbacks import ModelCheckpoint, Callback
# from pytorch_lightning.loggers import MLFlowLogger


# BASE_DIR = "/teamspace/studios/this_studio/gpt2tiny"
# DATA_CACHE_DIR = Path(BASE_DIR) / "data"

# mlflow_tracking_uri = "https://5000-01kgr6z0qq5h0srek4vj1jq4pb.cloudspaces.litng.ai"

# def main(
#     experiment_name,
#     run_name,
#     model_name,
#     tokenizer,
#     model_config,
#     trainer_config
# ):

#     mlf_logger = MLFlowLogger(
#         experiment_name=experiment_name,
#         tracking_uri=mlflow_tracking_uri,
#         run_name=run_name,
#         # artifact_location=f"{mlflow_tracking_uri}/artifacts", # you can explicitly set artifact_location if needed
#     )

#     # the mlflow run is started by MLFlowLogger internally when the trainer initializes.
#     # we don't need to manually start/stop it here if we're letting MLFlowLogger manage it,
#     # as it might interfere with the logger's lifecycle.
#     # if you want to ensure it's active for other mlflow calls outside the logger,
#     # you can wrap the entire main logic within an mlflow.start_run() after logger init.
#     # for now, let's remove the explicit mlflow.start_run() block for simplicity
#     # since MLFlowLogger handles it.

#     train_dataloader = DataLoader(
#         PreTokDataset(
#             model_config.block_size,
#             split="train",
#             data_dir=[DATA_CACHE_DIR / "TinyStories_all_data_only_pretrain"],
#             weights="Balanced",
#         ),
#         batch_size=trainer_config.batch_size,
#         num_workers=trainer_config.num_workers,
#     )

#     eval_dataloader = DataLoader(
#         PreTokDataset(
#             model_config.block_size,
#             split="validation",
#             data_dir=[DATA_CACHE_DIR / "TinyStories_all_data_only_pretrain"],
#             weights="Balanced",
#         ),
#         batch_size=trainer_config.batch_size,
#         num_workers=trainer_config.num_workers,
#     )


#     model = GPT2Module(
#         model_config,
#         tokenizer,
#         gen_every_n_epochs=500,
#         prompts=[
#             "A dragon in a cave",
#             "1+1 is",
#             "what is the gcd of 21 and 36?"
#         ]
#     )

#     # initialize ModelCheckpoint with a placeholder or None for dirpath
#     # the SetCheckpointDirCallback will update it later
#     checkpoint_cb = ModelCheckpoint(
#         monitor="val_loss",
#         mode="min",
#         save_top_k=1,
#         dirpath=None, # will be set by the callback
#         filename="best-{step}-{val_loss:.4f}",
#     )

#     trainer = pl.Trainer(
#         accelerator="auto",
#         devices=1,
#         precision="16-mixed",
#         max_steps=trainer_config.max_iters,
#         val_check_interval=trainer_config.eval_interval,
#         limit_val_batches=200,
#         logger=mlf_logger,
#         callbacks=[
#             checkpoint_cb,
#             LogBestCkptAndPyfuncToMLflow(module_cls=GPT2Module, register_name=model_name),
#             SetCheckpointDirCallback(checkpoint_cb) # add our new callback here
#         ],
#         log_every_n_steps=trainer_config.log_interval,
#         accumulate_grad_batches=trainer_config.gradient_accumulation_steps,
#         gradient_clip_val=trainer_config.grad_clip,
#     )


#     trainer.fit(model, train_dataloader, eval_dataloader)


# if __name__ == "__main__":
#     model_config = GPTConfig(flash=True)
#     trainer_config = TrainingConfig(batch_size=64, num_workers=4, max_iters=50)

#     experiment_name = "test"
#     run_name = "tinystories-pretrain"
#     run_name += "-" + pd.Timestamp.now().strftime("%Y-%m-%d-%H%M%S")

#     tokenizer = Tokenizer(f"{BASE_DIR}/data/tok4096_tinystories.model")
#     model_name = "GPT2Pretrained"
#     main(experiment_name, run_name, model_name, tokenizer, model_config, trainer_config)