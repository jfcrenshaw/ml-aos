"""Utility functions."""
import torch


def convert_zernikes(
    zernikes: torch.Tensor,
) -> torch.Tensor:
    """Convert zernike units from wavelengths to quadrature contribution to FWHM.

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
    return torch.mean(convert_zernikes(pred - true) ** 2, axis=1, keepdim=True)


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
