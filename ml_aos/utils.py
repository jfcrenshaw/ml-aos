"""Utility functions."""
import os
from pathlib import Path

import git
import matplotlib.pyplot as plt
import numpy as np
import torch


def convert_zernikes(
    zernikes: torch.Tensor,
) -> torch.Tensor:
    """Convert zernike units from microns to quadrature contribution to FWHM.

    Parameters
    ----------
    zernikes: torch.Tensor
        Tensor of zernike coefficients in microns.

    Returns
    -------
    torch.Tensor
        Zernike coefficients in units of quadrature contribution to FWHM.
    """
    # these conversion factors depend on telescope radius and obscuration
    # the numbers below are for the Rubin telescope; different numbers
    # are needed for Auxtel. For calculating these factors, see ts_phosim
    arcsec_per_micron = zernikes.new(
        [
            0.751,  # Z4
            0.271,  # Z5
            0.271,  # Z6
            0.819,  # Z7
            0.819,  # Z8
            0.396,  # Z9
            0.396,  # Z10
            1.679,  # Z11
            0.937,  # Z12
            0.937,  # Z13
            0.517,  # Z14
            0.517,  # Z15
            1.755,  # Z16
            1.755,  # Z17
            1.089,  # Z18
            1.089,  # Z19
            0.635,  # Z20
            0.635,  # Z21
            2.810,  # Z22
        ]
    )

    return zernikes * arcsec_per_micron


def plot_zernikes(z_pred: torch.Tensor, z_true: torch.Tensor) -> plt.Figure:
    """Plot true and predicted zernikes (up to 8).

    Parameters
    ----------
    z_pred: torch.Tensor
        2D Array of predicted Zernike coefficients
    z_true: torch.Tensor
        2D Array of true Zernike coefficients

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
        ax.plot(np.arange(4, 23), convert_zernikes(zt), label="True")
        ax.plot(np.arange(4, 23), convert_zernikes(zp), label="Predicted")
        ax.set(xticks=np.arange(4, 23, 2))

    axes[0, 0].legend()  # add legend to first panel

    # set axis labels
    for ax in axes[:, 0]:
        ax.set_ylabel("Arcsec FWHM")
    for ax in axes[1, :]:
        ax.set_xlabel("Zernike number (Noll)")

    return fig


def count_parameters(model: torch.nn.Module, trainable: bool = True) -> int:
    """Count the number of parameters in the model.

    Parameters
    ----------
    model: torch.nn.Module
        The Pytorch model to count parameters for
    trainable: bool, default=True
        If True, only counts trainable parameters

    Returns
    -------
    int
        The number of trainable parameters
    """
    if trainable:
        return sum(
            params.numel() for params in model.parameters() if params.requires_grad
        )
    else:
        return sum(params.numel() for params in model.parameters())


def printOnce(msg: str, header: bool = False) -> None:
    """Print message once to the terminal.

    This avoids the problem where statements get printed multiple times in
    a distributed setting.

    Parameters
    ----------
    msg: str
        Message to print
    header: bool, default=False
        Whether to add extra space and underline for the message
    """
    rank = os.environ.get("LOCAL_RANK", None)
    if rank is None or rank == "0":
        if header:
            msg = f"\n{msg}\n{'-'*len(msg)}\n"
        print(msg)


def transform_inputs(
    image: np.ndarray,
    fx: float,
    fy: float,
    intra: bool,
    band: int,
) -> tuple:
    """Transform inputs to the neural network.

    Parameters
    ----------
    image: np.ndarray
        The donut image
    fx: float
        X angle of source with respect to optic axis (radians)
    fy: float
        Y angle of source with respect to optic axis (radians)
    intra: bool
        Boolean indicating whether the donut is intra or extra focal
    band: int
        Band index in the string "ugrizy". I.e., 0="u", ..., 5="y".

    Returns
    -------
        same as above, with transformations applied
    """
    # rescale image to [0, 1]
    image -= image.min()
    image /= image.max()

    # normalize image
    image_mean = 0.347
    image_std = 0.226
    image = (image - image_mean) / image_std

    # normalize angles
    field_mean = 0.000
    field_std = 0.021
    fx = (fx - field_mean) / field_std
    fy = (fy - field_mean) / field_std

    # normalize the intrafocal flags
    intra_mean = 0.5
    intra_std = 0.5
    intra = (intra - intra_mean) / intra_std  # type: ignore

    # get the effective wavelength in microns
    band = {  # type: ignore
        0: 0.3671,
        1: 0.4827,
        2: 0.6223,
        3: 0.7546,
        4: 0.8691,
        5: 0.9712,
    }[band]

    # normalize the wavelength
    band_mean = 0.710
    band_std = 0.174
    band = (band - band_mean) / band_std  # type: ignore

    return image, fx, fy, intra, band


def get_root() -> Path:
    """Return the absolute path of the git root directory.

    Returns
    -------
    pathlib.PosixPath
    """
    root = Path(git.Repo(".", search_parent_directories=True).working_tree_dir)

    return root
