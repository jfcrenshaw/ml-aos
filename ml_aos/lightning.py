"""Wrapping everything for DavidNet in Pytorch Lightning."""

from typing import Tuple

import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
import wandb
from torch.utils.data import DataLoader
from torchmetrics.functional import mean_squared_error

from ml_aos.dataloader import Donuts
from ml_aos.david_net import DavidNet as TorchDavidNet


class DonutLoader(pl.LightningDataModule):
    """Pytorch Lightning wrapper for the simulated Donuts DataSet."""

    def __init__(
        self,
        background: bool = True,
        badpix: bool = True,
        dither: int = 5,
        max_blend: float = 0.50,
        mask_blends: bool = False,
        center_brightest: bool = True,
        nval: int = 2 ** 16,
        ntest: int = 2 ** 16,
        split_seed: int = 0,
        batch_size: int = 64,
        num_workers: int = 16,
        persistent_workers: bool = True,
        pin_memory: bool = True,
    ) -> None:
        """Load the simulated Donuts data.

        Parameters
        ----------
        mode: str, default="train"
            Which set to load. Options are train, val (i.e. validation),
            or test.
        background: bool, default=True
            Whether to add the sky background to the donut images.
        badpix: bool, default=True
            Whether to simulate bad pixels and columns.
        dither: int, default=5
            Maximum number of pixels to dither in both directions.
            This simulates mis-centering.
        max_blend: float, default=0.50
            Maximum fraction of the central star to be blended. For images
            with many blends, only the first handful of stars will be drawn,
            stopping when the next star would pass this blend threshold.
        mask_blends: bool, default=False
            Whether to mask the blends.
        center_brightest: bool, default=True
            Whether to center the brightest star in blended images.
        nval: int, default=256
            Number of donuts in the validation set.
        ntest: int, default=2048
            Number of donuts in the test set
        split_seed: int, default=0
            Random seed for training set/test set/validation set selection.
        batch_size: int, default=64
            The batch size for SGD.
        num_workers: int, default=16
            The number of workers for parallel loading of batches.
        persistent_workers: bool, default=True
            Whether to shutdown worker processes after dataset is consumed once
        pin_memory: bool, default=True
            Whether to automatically put data in pinned memory (recommended
            whenever using a GPU).
        """
        super().__init__()
        self.save_hyperparameters()

    def _build_loader(self, mode: str) -> DataLoader:
        return DataLoader(
            Donuts(
                mode=mode,
                background=self.hparams.background,
                badpix=self.hparams.badpix,
                dither=self.hparams.dither,
                max_blend=self.hparams.max_blend,
                mask_blends=self.hparams.mask_blends,
                center_brightest=self.hparams.center_brightest,
                nval=self.hparams.nval,
                ntest=self.hparams.ntest,
                split_seed=self.hparams.split_seed,
            ),
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            persistent_workers=self.hparams.persistent_workers,
            pin_memory=self.hparams.pin_memory,
        )

    def train_dataloader(self) -> DataLoader:
        """Return the training DataLoader."""
        return self._build_loader("train")

    def val_dataloader(self) -> DataLoader:
        """Return the validation DataLoader."""
        return self._build_loader("val")

    def test_dataloader(self) -> DataLoader:
        """Return the testing DataLoader."""
        return self._build_loader("test")


class DavidNet(TorchDavidNet, pl.LightningModule):
    """Pytorch Lightning wrapper for training DavidNet."""

    def __init__(self, n_meta_layers: int = 3) -> None:
        """Create the DavidNet.

        Parameters
        ----------
        n_meta_layers: int, default=3
            Number of layers in the MetaNet inside the DavidNet. These
            are the linear layers that map image features plus field
            position to Zernike coefficients.
        """
        # set up the DavidNet implemented in torch,
        # as well as the LightningModule boilerplate
        super().__init__(n_meta_layers=n_meta_layers)

        # save the hyperparams in the log
        self.save_hyperparameters()

    def _predict(
        self, batch: dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Make predictions for a batch of donuts."""
        # unpack the data
        img = batch["image"]
        z_true = batch["zernikes"]
        fx = batch["field_x"]
        fy = batch["field_y"]
        intra = batch["intrafocal"]

        # predict the zernikes
        z_pred = self(img, fx, fy, intra)

        # compute the MSE
        loss = mean_squared_error(z_pred, z_true)

        return z_true, z_pred, loss

    def training_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Calculate the loss of the training step."""
        # calculate the loss for this batch
        *_, loss = self._predict(batch)

        # save the loss in the log
        self.log("train_loss", loss)

        return loss

    def validation_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> None:
        """Perform validation step."""
        # predict for validation sample
        z_true, z_pred, loss = self._predict(batch)

        # log the loss
        self.log("val_loss", loss)

        # for the first batch of the validation set, plot the Zernikes
        if batch_idx == 0 and wandb.run is not None:
            # draw the Zernike figure and convert to wandb image for logging
            fig = wandb.Image(plot_zernikes(z_true.cpu(), z_pred.cpu()))
            # log the image
            wandb.log(
                {"zernikes": fig, "global_step": self.trainer.global_step}
            )


def plot_zernikes(z_true: torch.Tensor, z_pred: torch.Tensor) -> plt.Figure:
    """Plot true and predicted zernikes (up to 8).

    Parameters
    ----------
    z_true: torch.Tensor
        2D Array of true Zernike coefficients
    z_pred: torch.Tensor
        2D Array of predicted Zernike coefficients

    Returns
    -------
    plt.Figure
        Figure containing the 8 axes with the true and predicted Zernike
        coefficients plotted together.
    """
    # create the figure
    fig, axes = plt.subplots(
        2,
        4,
        figsize=(12, 5),
        constrained_layout=True,
        dpi=150,
        sharex=True,
        sharey=True,
    )

    # loop through the axes/zernikes
    for ax, zt, zp in zip(axes.flatten(), z_true, z_pred):
        ax.plot(zt, label="True")
        ax.plot(zp, label="Predicted")

    axes[0, 0].set(xticks=[])  # remove x ticks
    axes[0, 0].legend()  # add legend to first panel

    return fig
