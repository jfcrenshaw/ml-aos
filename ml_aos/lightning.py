"""Wrapping everything for WaveNet in Pytorch Lightning."""

from typing import Any

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from ml_aos.dataloader import Donuts
from ml_aos.utils import convert_zernikes
from ml_aos.wavenet import WaveNet


class DonutLoader(pl.LightningDataModule):
    """Pytorch Lightning wrapper for the simulated Donuts DataSet."""

    def __init__(
        self,
        batch_size: int = 64,
        num_workers: int = 16,
        persistent_workers: bool = True,
        pin_memory: bool = True,
        shuffle: bool = True,
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
        shuffle: bool, default=True
            Whether to shuffle the train dataloader.
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
        return self._build_loader("train", shuffle=self.hparams.shuffle)

    def val_dataloader(self) -> DataLoader:
        """Return the validation DataLoader."""
        return self._build_loader("val", drop_last=False)

    def test_dataloader(self) -> DataLoader:
        """Return the testing DataLoader."""
        return self._build_loader("test", drop_last=False)


class WaveNetSystem(pl.LightningModule):
    """Pytorch Lightning system for training the WaveNet."""

    def __init__(
        self,
        cnn_model: str = "resnet18",
        freeze_cnn: bool = False,
        n_predictor_layers: tuple = (256,),
        lr: float = 1e-3,
        lr_schedule: bool = False,
    ) -> None:
        """Create the WaveNet.

        Parameters
        ----------
        cnn_model: str, default="resnet18"
            The name of the pre-trained CNN model from torchvision.
        freeze_cnn: bool, default=False
            Whether to freeze the CNN weights.
        n_predictor_layers: tuple, default=(256)
            Number of nodes in the hidden layers of the Zernike predictor network.
            This does not include the output layer, which is fixed to 19.
        lr: float, default=1e-3
            The initial learning rate for Adam.
        lr_schedule: bool, default=True
            Whether to use the ReduceLROnPlateau learning rate scheduler.
        """
        super().__init__()
        self.save_hyperparameters()
        self.wavenet = WaveNet(
            cnn_model=cnn_model,
            n_predictor_layers=n_predictor_layers,
        )

        # define some parameters that will be accessed by
        # the MachineLearningAlgorithm in ts_wep
        self.camType = "LsstCam"
        self.inputShape = (170, 170)

    def predict_step(
        self, batch: dict, batch_idx: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict Zernikes and return with truth."""
        # unpack data from the dictionary
        img = batch["image"]
        fx = batch["field_x"]
        fy = batch["field_y"]
        intra = batch["intrafocal"]
        band = batch["band"]
        zk_true = batch["zernikes"]
        dof_true = batch["dof"]  # noqa: F841

        # predict zernikes
        zk_pred = self.wavenet(img, fx, fy, intra, band)

        return zk_pred, zk_true

    def calc_losses(self, batch: dict, batch_idx: int) -> tuple:
        """Predict Zernikes and calculate the losses.

        The two losses considered are:
        - mSSE - mean of the SSE (in arcsec^2)
        - mRSSE - mean of the root of the SSE (in arcsec)
        where SSE = Sum of Squared Errors, and the mean is taken over the batch

        The mRSSE provides an estimate of the PSF degradation.
        """
        # predict zernikes
        zk_pred, zk_true = self.predict_step(batch, batch_idx)

        # convert to FWHM contributions
        zk_pred = convert_zernikes(zk_pred)
        zk_true = convert_zernikes(zk_true)

        # calculate loss
        sse = F.mse_loss(zk_pred, zk_true, reduction="none").sum(dim=-1)
        mSSE = sse.mean()
        mRSSE = torch.sqrt(sse).mean()

        return mSSE, mRSSE

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        """Execute training step on a batch."""
        mSSE, mRSSE = self.calc_losses(batch, batch_idx)
        self.log("train_mSSE", mSSE, sync_dist=True, prog_bar=True)
        self.log("train_mRSSE", mRSSE, sync_dist=True)

        return mSSE

    def validation_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        """Execute validation step on a batch."""
        mSSE, mRSSE = self.calc_losses(batch, batch_idx)
        self.log("val_mSSE", mSSE, sync_dist=True, prog_bar=True)
        self.log("val_mRSSE", mRSSE, sync_dist=True)

        return mSSE

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure the optimizer."""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

        if self.hparams.lr_schedule:
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": ReduceLROnPlateau(optimizer),
                    "monitor": "val_mSSE",
                    "frequency": 1,
                },
            }
        else:
            return optimizer

    def forward(
        self,
        img: torch.Tensor,
        fx: torch.Tensor,
        fy: torch.Tensor,
        focalFlag: torch.Tensor,
        band: torch.Tensor,
    ) -> torch.Tensor:
        """Predict zernikes for production.

        This method assumes the inputs have NOT been previously
        transformed by ml_aos.utils.transform_inputs.
        """
        # rescale image to [0, 1]
        img -= img.min()
        img /= img.max()

        # normalize image
        image_mean = 0.347
        image_std = 0.226
        img = (img - image_mean) / image_std

        # convert angles to radians
        fx *= torch.pi / 180
        fy *= torch.pi / 180

        # normalize angles
        field_mean = 0.000
        field_std = 0.021
        fx = (fx - field_mean) / field_std
        fy = (fy - field_mean) / field_std

        # normalize the intrafocal flags
        intra_mean = 0.5
        intra_std = 0.5
        focalFlag = (focalFlag - intra_mean) / intra_std

        # get the effective wavelength in microns
        band = {
            0: torch.FloatTensor([[0.3671]]),
            1: torch.FloatTensor([[0.4827]]),
            2: torch.FloatTensor([[0.6223]]),
            3: torch.FloatTensor([[0.7546]]),
            4: torch.FloatTensor([[0.8691]]),
            5: torch.FloatTensor([[0.9712]]),
        }[band.item()]

        # normalize the wavelength
        band_mean = 0.710
        band_std = 0.174
        band = (band - band_mean) / band_std

        # predict zernikes in microns
        zk_pred = self.wavenet(img, fx, fy, focalFlag, band)

        # convert to nanometers
        zk_pred *= 1_000

        return zk_pred
