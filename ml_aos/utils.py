"""Utility functions."""
import os

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


def get_corner(fx: float, fy: float) -> int:
    """Maps a field position to a corner number.

    Parameters
    ----------
    fx: float
        X angle of source with respect to optic axis (radians)
    fy: float
        Y angle of source with respect to optic axis (radians)

    Returns
    -------
    int
        Corner number. Top right corner is zero, increasing counter-clockwise

    """
    # determine which corner we are in
    if fx > 0 and fy > 0:
        corner = 0
    elif fx < 0 and fy > 0:
        corner = 1
    elif fx < 0 and fy < 0:
        corner = 2
    else:
        corner = 3

    return corner


def transform_inputs(
    image: torch.Tensor,
    fx: torch.Tensor,
    fy: torch.Tensor,
    intra: torch.Tensor,
    wavelen: torch.Tensor,
) -> tuple:
    """Transform the inputs to the neural network.

    Parameters
    ----------
    image: torch.Tensor
        The donut image
    fx: torch.Tensor
        X angle of source with respect to optic axis (radians)
    fy: torch.Tensor
        Y angle of source with respect to optic axis (radians)
    intra: torch.Tensor
        Boolean indicating whether the donut is intra or extra focal
    wavelen: torch.Tensor
        Effective wavelength of observation in microns.

    Returns
    -------
    Tuple[torch.Tensor x 5]
        same as above, with transformations applied
    torch.Tensor
        One-hot-encoding of the corner
    """
    # create a one-hot-encoding for the corner
    corner = get_corner(fx, fy)
    corner_one_hot = torch.zeros(4)
    corner_one_hot[corner] = 1

    # take absolute value of the angles
    fx, fy = torch.abs(fx), torch.abs(fy)

    # for odd corners, swap fx and fy
    if corner in [1, 3]:
        fx, fy = fy, fx

    # normalize angles
    field_mean = 0.021
    field_std = 0.001
    fx = (fx - field_mean) / field_std
    fy = (fy - field_mean) / field_std

    # rotate the image to corner 0
    image = torch.rot90(image, corner)

    # rescale image to [0, 1]
    image -= image.min()
    image /= image.max()

    # normalize image
    image_mean = 0.347
    image_std = 0.226
    image = (image - image_mean) / image_std

    # normalize the intrafocal flags
    intra_mean = 0.5
    intra_std = 0.5
    intra = (intra - intra_mean) / intra_std

    # normalize the wavelength
    wavelen_mean = 0.710
    wavelen_std = 0.174
    wavelen = (wavelen - wavelen_mean) / wavelen_std

    return image, fx, fy, intra, wavelen, corner_one_hot
