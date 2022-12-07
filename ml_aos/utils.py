"""Utility functions."""
import numpy as np
import numpy.typing as npt


def convert_zernikes(
    zernikes: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Convert zernike units from wavelengths to quadrature contribution to FWHM.

    Parameters
    ----------
    zernikes: np.ndarray
        Array of zernike coefficients in microns.

    Returns
    -------
    np.ndarray
        Zernike coefficients in units of quadrature contribution to FWHM.
    """
    # these conversion factors depend on telescope radius and obscuration
    # the numbers below are for the Rubin telescope; different numbers
    # are needed for Auxtel. For calculating these factors, see ts_phosim
    arcsec_per_micron = np.array(
        [  # type: ignore
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
