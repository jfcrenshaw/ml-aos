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
        n_meta_layers: int = 2,
        n_meta_nodes: int = 16,
        n_predictor_layers: tuple = (256,),
        lr: float = 1e-3,
    ) -> None:
        """Create the WaveNet.

        Parameters
        ----------
        cnn_model: str, default="resnet18"
            The name of the pre-trained CNN model from torchvision.
        n_meta_layers: int, default=2
            Number of layers in the MetaNet, including the output layer.
        n_meta_nodes: int, default=16
            Number of nodes in each layer of the MetaNet.
        n_predictor_layers: tuple, default=(256)
            Number of nodes in the hidden layers of the Zernike predictor network.
            This does not include the output layer, which is fixed to 19.
        lr: float, default=1e-3
            The initial learning rate for Adam.
        """
        super().__init__()
        self.save_hyperparameters()
        self.wavenet = WaveNet(
            cnn_model=cnn_model,
            n_meta_layers=n_meta_layers,
            n_meta_nodes=n_meta_nodes,
            n_predictor_layers=n_predictor_layers,
        )

    def predict_step(
        self, batch: dict, batch_idx: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict Zernikes and return with truth."""
        # unpack data from the dictionary
        img = batch["image"]
        fx = batch["field_x"]
        fy = batch["field_y"]
        intra = batch["intrafocal"]
        corner = batch["corner"]
        wavelen = batch["wavelen"]
        zk_true = batch["zernikes"]
        dof_true = batch["dof"]  # noqa: F841

        # predict zernikes
        zk_pred = self.wavenet(img, fx, fy, intra, corner, wavelen)

        return zk_pred, zk_true

    def calculate_loss(self, batch: dict, batch_idx: int) -> tuple:
        """Predict Zernikes and calculate the loss."""
        # predict zernikes
        zk_pred, zk_true = self.predict_step(batch, batch_idx)

        # convert to FWHM contributions
        zk_pred = convert_zernikes(zk_pred)
        zk_true = convert_zernikes(zk_true)

        # calculate loss
        loss = F.mse_loss(zk_pred, zk_true)

        return loss

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        """Execute training step on a batch."""
        loss = self.calculate_loss(batch, batch_idx)
        self.log("train_loss", loss, sync_dist=True, prog_bar=True)

        return loss

    def validation_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        """Execute validation step on a batch."""
        loss = self.calculate_loss(batch, batch_idx)
        self.log("val_loss", loss, sync_dist=True, prog_bar=True)

        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure the optimizer."""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": ReduceLROnPlateau(optimizer),
                "monitor": "val_loss",
                "frequency": 1,
            },
        }
        return optimizer

    def forward(
        self,
    ) -> None:
        """Predict zernikes for production."""
        # need to take the original image, field angles, intra flag, corner, and band
        # then transform these, including getting wavelength from the band
        # then reshape the image
        # then feed into the network
        # then make sure the output units are what we want.
        pass
