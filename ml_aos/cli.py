"""Script for running WaveNet from the command line using Lightning CLI."""
import os

import pytorch_lightning as pl
from pytorch_lightning.cli import LightningCLI

from ml_aos.lightning import WaveNet, DonutLoader

if __name__ == "__main__":

    # setup the wandb logger
    wandb_logger = pl.loggers.WandbLogger(
        log_model="all",
    )

    # export wandb name so other GPUs can see the name too!
    if os.environ.get("LOCAL_RANK", None) is None:
        os.environ["WANDB_NAME"] = wandb_logger.experiment.name
    wandb_logger._save_dir = os.environ["WANDB_NAME"]
    wandb_logger.log_dir = os.environ["WANDB_NAME"]

    # setup the CLI
    cli = LightningCLI(
        WaveNet,
        DonutLoader,
        trainer_defaults={"logger": wandb_logger},
    )
