"""Wrapping everything for DavidNet in Pytorch Lightning."""

from typing import Tuple

import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
import wandb
from torch.utils.data import DataLoader

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
        normalize_pixels: bool = True,
        convert_zernikes: bool = True,
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
        normalize_pixels: bool, default=True
            Whether to normalize the pixel values using the mean and std
            of the single-donut pixels.
        convert_zernikes: bool, default=True
            Whether to convert Zernike coefficients from units of r band
            wavelength to quadrature contribution to PSF FWHM.
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

    def _build_loader(self, mode: str, shuffle: bool = False) -> DataLoader:
        return DataLoader(
            Donuts(
                mode=mode,
                background=self.hparams.background,
                badpix=self.hparams.badpix,
                dither=self.hparams.dither,
                max_blend=self.hparams.max_blend,
                mask_blends=self.hparams.mask_blends,
                center_brightest=self.hparams.center_brightest,
                normalize_pixels=self.hparams.normalize_pixels,
                convert_zernikes=self.hparams.convert_zernikes,
                nval=self.hparams.nval,
                ntest=self.hparams.ntest,
                split_seed=self.hparams.split_seed,
            ),
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            persistent_workers=self.hparams.persistent_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=shuffle,
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
        self, batch: dict[str, torch.Tensor], batch_idx: int
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
        self, batch: dict[str, torch.Tensor], batch_idx: int
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
            wandb.log(
                {"zernikes": fig, "global_step": self.trainer.global_step}
            )
            del fig

        # calculate distance from the center of focal plane in meters
        x = batch["focal_x"]
        y = batch["focal_y"]
        dist_rads = torch.sqrt(x ** 2 + y ** 2)  # distance in radians
        dist_arcsecs = dist_rads * 206_265  # distance in arcsecs
        dist_microns = dist_arcsecs * 5  # distance in microns
        dist_meters = dist_microns / 1e6

        # get the fraction blended
        frac_blended = batch["fraction_blended"]

        val_outputs = torch.hstack((dist_meters, frac_blended, mse))

        return val_outputs

    def validation_epoch_end(self, val_outputs: torch.Tensor) -> None:
        """Compute metrics for the whole validation epoch."""
        # extract the validation outputs
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
        self.log("val_loss_blended", mse[blended_idx].mean())


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

    # set axis labels
    for ax in axes[:, 0]:
        ax.set_ylabel("Arcsec FWHM")
    for ax in axes[1, :]:
        ax.set_xlabel("Zernike number (Noll)")

    return fig


def plot_loss_vs_blended(
    frac_blended: torch.Tensor, val_loss: torch.Tensor
) -> plt.Figure:
    """Plot validation loss vs the fraction blended

    Parameters
    ----------
    frac_blended: torch.Tensor
        Array of blend fraction.
    val_loss: torch.Tensor
        Array of validation losses

    Returns
    -------
    plt.Figure
        Figure containing plot of validation loss vs fraction blended
    """
    fig, ax = plt.subplots(constrained_layout=True, dpi=150)
    ax.scatter(frac_blended[:100], val_loss[:100], marker=".", rasterized=True)
    ax.set(xlabel="Fraction blended", ylabel="Validation loss [arcsec FWHM]")
    return fig


def plot_loss_vs_distance(
    distance: torch.Tensor, val_loss: torch.Tensor
) -> plt.Figure:
    """Plot validation loss vs the distance from the center of the focal
    plane, in meters.

    Parameters
    ----------
    distance: torch.Tensor
        Array of distances from the center of the focal plane, in meters
    val_loss: torch.Tensor
        Array of validation losses

    Returns
    -------
    plt.Figure
        Figure containing plot of validation loss vs distance from center
        of the focal plane
    """
    fig, ax = plt.subplots(constrained_layout=True, dpi=150)
    ax.scatter(distance, val_loss, marker=".")
    ax.set(
        xlabel="Dist. from center of focal plane [m]",
        ylabel="Validation loss [arcsec FWHM]",
    )
    return fig
