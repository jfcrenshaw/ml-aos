"""Wrapping everything for WaveNet in Pytorch Lightning."""

from typing import Any, Dict, Tuple

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

import wandb
from ml_aos.dataloader import Donuts
from ml_aos.wave_net import WaveNet as TorchWaveNet
from ml_aos.plotting import plot_zernikes
from ml_aos.utils import calc_mse


class DonutLoader(pl.LightningDataModule):
    """Pytorch Lightning wrapper for the simulated Donuts DataSet."""

    def __init__(
        self,
        batch_size: int = 64,
        num_workers: int = 16,
        persistent_workers: bool = True,
        pin_memory: bool = True,
        **kwargs: Any,
    ) -> None:
        """Load the simulated Donuts data.

        Parameters
        ----------
        batch_size: int, default=64
            The batch size for SGD.
        num_workers: int, default=16
            The number of workers for parallel loading of batches.
        persistent_workers: bool, default=True
            Whether to shutdown worker processes after dataset is consumed once
        pin_memory: bool, default=True
            Whether to automatically put data in pinned memory (recommended
            whenever using a GPU).
        **kwargs
            See the keyword arguments in the Donuts class.
        """
        super().__init__()
        self.save_hyperparameters()

    def _build_loader(
        self, mode: str, shuffle: bool = False, drop_last: bool = True
    ) -> DataLoader:
        """Build a DataLoader"""
        return DataLoader(
            Donuts(mode=mode, **self.hparams),
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            persistent_workers=self.hparams.persistent_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=shuffle,
            drop_last=drop_last,
        )

    def train_dataloader(self) -> DataLoader:
        """Return the training DataLoader."""
        return self._build_loader("train", shuffle=True)

    def val_dataloader(self) -> DataLoader:
        """Return the validation DataLoader."""
        return self._build_loader("val", drop_last=False)

    def test_dataloader(self) -> DataLoader:
        """Return the testing DataLoader."""
        return self._build_loader("test", drop_last=False)


class WaveNet(TorchWaveNet, pl.LightningModule):
    """Pytorch Lightning wrapper for WaveNet."""

    def __init__(self, n_meta_layers: int = 3) -> None:
        """Create the WaveNet.

        Parameters
        ----------
        n_meta_layers: int, default=3
            Number of layers in the MetaNet inside the WaveNet. These
            are the linear layers that map image features plus field
            position to Zernike coefficients.
        """
        # set up the WaveNet implemented in torch,
        # as well as the LightningModule boilerplate
        super().__init__(n_meta_layers=n_meta_layers)

        # save the hyperparams in the log
        self.save_hyperparameters()

    def _predict(
        self, batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Make predictions for a batch of donuts."""
        # unpack the data
        img = batch["image"]
        z_true = batch["zernikes"]
        fx = batch["field_x"]
        fy = batch["field_y"]
        intra = batch["intrafocal"]

        # predict the zernikes
        z_pred = self(img, fx, fy, intra)

        return z_pred, z_true

    def training_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Calculate the loss of the training step."""
        # calculate the MSE for the batch
        z_pred, z_true = self._predict(batch)
        mse = calc_mse(z_pred, z_true)

        # log the mean rmse
        self.log("train_rmse", torch.sqrt(mse).mean())

        # loss = mean mse
        loss = mse.mean()
        self.log("train_loss", loss)

        return loss

    def validation_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> Tuple[torch.Tensor, ...]:
        """Perform validation step."""
        # calculate the MSE for the validation sample
        z_pred, z_true = self._predict(batch)
        mse = calc_mse(z_pred, z_true)

        # log the mean rmse
        self.log("val_rmse", torch.sqrt(mse).mean())

        # log the loss
        self.log("val_loss", mse.mean())

        # for the first batch of the validation set, plot the Zernikes
        if batch_idx == 0 and wandb.run is not None:
            # draw the Zernike figure and convert to wandb image for logging
            fig = wandb.Image(plot_zernikes(z_true.cpu(), z_pred.cpu()))
            # log the image
            wandb.log({"zernikes": fig, "global_step": self.trainer.global_step})
            del fig

        val_outputs = mse

        return val_outputs

    def validation_epoch_end(self, val_outputs: torch.Tensor) -> None:
        """Compute metrics for the whole validation epoch."""

        mse = torch.stack(val_outputs)
        self.log("val_loss", mse.mean())
        self.log("val_rmse", torch.sqrt(mse).mean())

    @torch.jit.export
    def tswep_predict(
        self,
        img: torch.Tensor,
        fx: torch.Tensor,
        fy: torch.Tensor,
        focalFlag: torch.Tensor,
    ) -> torch.Tensor:
        """Predict Zernikes for a CompensableImage from ts_wep.

        Parameters
        ----------
        img: torch.Tensor
            The donut image
        fx: torch.Tensor
            x-axis field angle, in degrees
        fy: torch.Tensor
            y-axis field angle, in degrees
        focalFlag: torch.Tensor
            Float indicating whether the donut is intra (1.) or extra (2.) focal

        Returns
        -------
        torch.Tensor
            Tensor of Zernike coefficients
        """
        # convert the field angles to radians
        fx = torch.deg2rad(fx)
        fy = torch.deg2rad(fy)

        # predict the zernikes in microns
        z_pred = self(img, fx, fy, focalFlag)

        # convert from microns to nanometers
        z_pred *= 1e3

        return z_pred
