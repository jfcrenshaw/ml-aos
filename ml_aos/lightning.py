"""Wrapping everything for DavidNet in Pytorch Lightning."""

from typing import Any, Dict, Tuple

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

import wandb
from ml_aos.dataloader import DavidsDonuts, JFsDonuts
from ml_aos.david_net import DavidNet as TorchDavidNet
from ml_aos.plotting import plot_zernikes


class DonutLoader(pl.LightningDataModule):
    """Pytorch Lightning wrapper for the simulated Donuts DataSet."""

    def __init__(
        self,
        sims: str = "JF",
        batch_size: int = 64,
        num_workers: int = 16,
        persistent_workers: bool = True,
        pin_memory: bool = True,
        **kwargs: Any,
    ) -> None:
        """Load the simulated Donuts data.

        Parameters
        ----------
        sims: str, default="JF"
            Which set of simulations to use. Either "JF" or "David".
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
            See the keyword arguments in the two DataLoader Classes.
        """
        super().__init__()
        self.save_hyperparameters()
        if sims == "JF":
            self._donut_loader = JFsDonuts
        elif sims == "David":
            self._donut_loader = DavidsDonuts  # type: ignore
        else:
            raise ValueError("sims must be 'JF' or 'David'.")

    def _build_loader(self, mode: str, shuffle: bool = False) -> DataLoader:
        return DataLoader(
            self._donut_loader(mode, **self.hparams),
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            persistent_workers=self.hparams.persistent_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=shuffle,
            drop_last=True,
        )

    def train_dataloader(self) -> DataLoader:
        """Return the training DataLoader."""
        return self._build_loader("train", shuffle=True)

    def val_dataloader(self) -> DataLoader:
        """Return the validation DataLoader."""
        return self._build_loader("val")

    def test_dataloader(self) -> DataLoader:
        """Return the testing DataLoader."""
        return self._build_loader("test")


class DavidNet(TorchDavidNet, pl.LightningModule):
    """Pytorch Lightning wrapper for training DavidNet."""

    def __init__(self, n_meta_layers: int = 3, input_shape: int = 256) -> None:
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
        super().__init__(n_meta_layers=n_meta_layers, input_shape=input_shape)

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

        """# calculate distance from the center of focal plane in meters
        x = batch["focal_x"]
        y = batch["focal_y"]
        dist_rads = torch.sqrt(x ** 2 + y ** 2)  # distance in radians
        dist_arcsecs = dist_rads * 206_265  # distance in arcsecs
        dist_microns = dist_arcsecs * 5  # distance in microns
        dist_meters = dist_microns / 1e6

        # get the fraction blended
        frac_blended = batch["fraction_blended"]

        val_outputs = torch.hstack((dist_meters, frac_blended, mse))"""

        val_outputs = mse

        return val_outputs

    def validation_epoch_end(self, val_outputs: torch.Tensor) -> None:
        """Compute metrics for the whole validation epoch."""
        """ # extract the validation outputs
        val_outputs = torch.stack(val_outputs).reshape(-1, 3)
        frac_blended = val_outputs[:, 1]
        mse = val_outputs[:, 2]

        # compute the validation loss for the unblended stars
        unblended_idx = torch.where(frac_blended < 0.01)
        self.log("val_rmse_unblended", torch.sqrt(mse[unblended_idx]).mean())
        self.log("val_loss_unblended", mse[unblended_idx].mean())

        # compute the validation loss for the blended stars
        blended_idx = torch.where(frac_blended >= 0.01)
        self.log("val_rmse_blended", torch.sqrt(mse[blended_idx]).mean())
        self.log("val_loss_blended", mse[blended_idx].mean())"""

        mse = torch.stack(val_outputs)
        self.log("val_loss", mse.mean())
        self.log("val_rmse", torch.sqrt(mse).mean())


def calc_mse(pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
    """Calculate the MSE for the predicted values.

    Parameters
    ----------
    pred: torch.Tensor
        Array of predicted values
    true: torch.Tensor
        Array of true values

    Returns
    -------
    torch.Tensor
        Array of MSE values
    """
    return torch.mean((pred - true) ** 2, axis=1, keepdim=True)
