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
        Array of zernike coefficients in units of r-band wavelength.

    Returns
    -------
    np.ndarray
        Zernike coefficients in units of quadrature contribution to FWHM.
    """

    rband_eff_wave = 0.6173  # microns

    # these conversion factors depend on telescope radius and obscuration
    # the numbers below are for the Rubin telescope; different numbers
    # are needed for Auxtel. Source: Josh Meyers
    arcsec_per_micron = np.array(  # type: ignore
        [
            1.062,  # Z4
            0.384,  # Z5
            0.384,  # Z6
            1.159,  # Z7
            1.159,  # Z8
            0.560,  # Z9
            0.560,  # Z10
            2.375,  # Z11
            1.325,  # Z12
            1.325,  # Z13
            0.730,  # Z14
            0.730,  # Z15
            2.482,  # Z16
            2.482,  # Z17
            1.541,  # Z18
            1.541,  # Z19
            0.898,  # Z20
            0.898,  # Z21
        ],
    )

    return zernikes * rband_eff_wave * arcsec_per_micron
