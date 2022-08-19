"""Functions for plotting metrics and results."""
import matplotlib.pyplot as plt
import torch


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
    """Plot validation loss vs the fraction blended.

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
    """Plot validation loss vs distance from center of focal plane, in meters.

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
