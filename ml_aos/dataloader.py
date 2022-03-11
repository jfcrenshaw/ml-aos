"""Pytorch DataSet for the AOS simulations.

Based on code written by David Thomas for his PhD Thesis at Stanford.
"""

import galsim
import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
from scipy import ndimage
from torch.utils.data import Dataset


class Donuts(Dataset):
    """Data set of AOS donuts and zernikes."""

    # number of blended and unblended simulations in the full set
    _N_UNBLENDED = 500_000
    _N_BLENDED = 100_404

    # estimated mean and std of pixel values for individual donuts
    PIXEL_MEAN = 30.31
    PIXEL_STD = 87.66

    # properties of the telescope
    PLATE_SCALE = 1 / 5  # arcsecs / micron
    PIXEL_SIZE = 10  # microns

    # angular conversion
    ARCSECS_PER_RADIAN = 206_265
    PIXELS_PER_RADIAN = ARCSECS_PER_RADIAN / PLATE_SCALE / PIXEL_SIZE

    def __init__(
        self,
        mode: str = "train",
        background: bool = True,
        badpix: bool = True,
        dither: int = 5,
        max_blend: float = 0.50,
        center_brightest: bool = True,
        normalize_pixels: bool = True,
        convert_zernikes: bool = True,
        mask_buffer: int = 0,
        nval: int = 2 ** 16,
        ntest: int = 2 ** 16,
        split_seed: int = 0,
        data_dir: str = "/epyc/users/jfc20/thomas_aos_sims/",
    ):
        """Load the simulated AOS donuts and zernikes in a Pytorch Dataset.

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
        center_brightest: bool, default=True
            Whether to center the brightest star in blended images.
        normalize_pixels: bool, default=True
            Whether to normalize the pixel values using the mean and std
            of the single-donut pixels.
        convert_zernikes: bool, default=True
            Whether to convert Zernike coefficients from units of r band
            wavelength to quadrature contribution to PSF FWHM.
        mask_buffer: int, default=0
            The number of buffer pixels to add to outside of masks.
        nval: int, default=256
            Number of donuts in the validation set.
        ntest: int, default=2048
            Number of donuts in the test set
        split_seed: int, default=0
            Random seed for training set/test set/validation set selection.
        data_dir: str, default=/epyc/users/jfc20/thomas_aos_sims/
            Location of the data directory. The default location is where
            I stored the simulations on epyc.
        """
        # check that the mode is valid
        allowed_modes = ["train", "val", "test"]
        if mode not in allowed_modes:
            raise ValueError(
                f"mode must be one of {', '.join(allowed_modes)}."
            )

        # save the data directory
        self.DATA_DIR = data_dir

        # set the image properties
        self.settings = {
            "background": background,
            "badpix": badpix,
            "dither": dither,
            "max_blend": max_blend,
            "center_brightest": center_brightest,
            "normalize_pixels": normalize_pixels,
            "convert_zernikes": convert_zernikes,
            "mask_buffer": mask_buffer,
        }

        # determine the indices of the val and test sets
        rng = np.random.default_rng(split_seed)
        holdout = rng.choice(
            self._N_UNBLENDED + self._N_BLENDED, nval + ntest, replace=False
        )

        # get the indices corresponding to the requested mode
        self.mode = mode
        if mode == "train":
            all_index = np.arange(self._N_UNBLENDED + self._N_BLENDED)
            train_index = set(all_index) - set(holdout)
            index = np.array(list(train_index))  # type: npt.NDArray[np.int64]
        elif mode == "val":
            index = holdout[:nval]
        else:
            index = holdout[nval:]

        # get the unblended images in this set
        unblended_idx = index[index < self._N_UNBLENDED]
        self.N_unblended = len(unblended_idx)

        # get the blended images in this set
        blended_idx = index[index >= self._N_UNBLENDED] - self._N_UNBLENDED
        self.N_blended = len(blended_idx)

        # load metadata for the unblended images
        unblended_df = pd.read_csv(self.DATA_DIR + "unblended/record.csv")
        unblended_df = unblended_df[np.isin(unblended_df.idx, unblended_idx)]
        self.unblended_df = unblended_df.reset_index(drop=True)

        # load metadata for the blended images
        blended_df = pd.read_csv(self.DATA_DIR + "blended/record.csv")
        blended_df = blended_df[np.isin(blended_df.idx, blended_idx)]
        self.blended_df = blended_df.set_index(
            blended_df.groupby("idx").ngroup() + self.N_unblended
        )

        # load the sky simulations
        self.sky = np.loadtxt(self.DATA_DIR + "sky.csv")

    def __len__(self) -> int:
        """Return length of this Dataset."""
        return len(self.unblended_df) + len(set(self.blended_df.idx))

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Return simulation corresponding to the index.

        Parameters
        ----------
        idx: int
            The index of the simulation to return.
            Valid values are [0, len(self) - 1], inclusive.

        Returns
        -------
        dict
            The dictionary contains the following pytorch tensors
                image: donut image, shape=(256, 256)
                mask: donut mask, shape=(256, 256)
                field_x, field_y: the field location in radians
                focal_x, focal_y: the focal plane position in radians
                intrafocal: boolean flag. 0 = extrafocal, 1 = intrafocal
                zernikes: Noll zernikes coefficients 4-21, inclusive
                n_blends: the number of blending stars in the image
                fraction_blended: fraction of the central donut blended
        """
        if idx < 0 or idx > len(self):
            raise ValueError("idx out of bounds.")

        # determine the amount of dithering
        dither = self.settings["dither"]  # max size of dither
        dx, dy = np.random.randint(-dither, dither + 1, size=2)

        # get the simulations for this index
        if idx < self.N_unblended:
            img, mask, fx, fy, px, py, intra, zernikes = self._get_unblended(
                idx, dx, dy
            )
            nb = 0  # n_blends=0
            fb = 0.0  # fraction_blended=0
        else:
            (
                img,
                mask,
                fx,
                fy,
                px,
                py,
                intra,
                zernikes,
                nb,
                fb,
            ) = self._get_blended(idx, dx, dy)

        # sky background and bad pixels (if requested)
        settings = self.settings
        img = self._apply_sky(img, seed=idx) if settings["background"] else img
        if settings["badpix"]:
            self._apply_badpix(img, mask, seed=idx)

        # reshape the images so they have a channel index
        img = img.reshape(1, 256, 256)

        # normalize the pixel values using the pixel mean and pixel std
        if self.settings["normalize_pixels"]:
            img = (img - self.PIXEL_MEAN) / self.PIXEL_STD

        # convert zernikes to their contribution to FWHM
        if self.settings["convert_zernikes"]:
            zernikes = self._convert_zernikes(zernikes)

        # convert outputs to torch tensors and return in a dictionary
        output = {
            "image": torch.from_numpy(img).float(),
            "mask": torch.from_numpy(mask).bool(),
            "field_x": torch.FloatTensor([fx]),
            "field_y": torch.FloatTensor([fy]),
            "focal_x": torch.FloatTensor([px]),
            "focal_y": torch.FloatTensor([py]),
            "intrafocal": torch.FloatTensor([intra]),
            "zernikes": torch.from_numpy(zernikes).float(),
            "n_blends": torch.ByteTensor([nb]),
            "fraction_blended": torch.FloatTensor([fb]),
        }

        return output

    def _get_unblended(
        self,
        idx: int,
        dx: int,
        dy: int,
        blend_idx: int = None,
    ) -> tuple[
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        np.float64,
        np.float64,
        np.float64,
        np.float64,
        bool,
        npt.NDArray[np.float64],
    ]:
        """Return an unblended simulation.

        Parameters
        ----------
        idx: int
            The index of the simulation. These refer to the pandas indices
            listed in self.unblended_df and self.blended_df.
        dx: int
            The amount to dither in the x direction
        dy: int
            The amount to dither in the y direction
        blend_idx: int, optional
            The neighborId lists in self.blended_df.

        Returns
        -------
        img: np.ndarray, shape=(256, 256)
            centered donut image
        mask: np.ndarray, shape=(256, 256)
            boolean array masking out background
        fx, fy: float
            the field location in radians
        px, py: float
            the focal plane position in radians
        intrafocal: bool
            True = intrafocal, False = extrafocal
        zernikes: np.ndarray, shape=(18,)
            Noll zernikes coefficients 4-21, inclusive
        """
        if blend_idx is None:
            file_idx = self.unblended_df.loc[idx].idx
            img_file = self.DATA_DIR + f"unblended/{file_idx}.image"
            zernike_file = self.DATA_DIR + f"unblended/{file_idx}.zernike"
            metadata = self.unblended_df.loc[idx]
        else:
            file_idx = self.blended_df.loc[idx].iloc[0].idx
            img_file = self.DATA_DIR + f"blended/{file_idx}_{blend_idx}.image"
            zernike_file = self.DATA_DIR + f"blended/{file_idx}_0.zernike"
            metadata = self.blended_df.loc[idx].iloc[blend_idx]

        # load the image array
        img = np.load(img_file)

        # create the donut mask
        mask = self._get_mask(img)

        # apply dither
        img = self._dither_img(img, dx, dy)

        # load the field position, in radians
        fx, fy = metadata.fieldx, metadata.fieldy

        # load the focal plane position, in radians
        px, py = metadata.posx, metadata.posy

        # boolean stating whether the image is intrafocal
        # if false, the image is extrafocal
        intra = "SW0" in metadata.chip

        # load the zernike coefficients; we use 4-21, inclusive
        zernikes = np.load(zernike_file)[4:22]

        return img, mask, fx, fy, px, py, intra, zernikes

    def _get_blended(
        self,
        idx: int,
        dx: int,
        dy: int,
    ) -> tuple[
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        np.float64,
        np.float64,
        np.float64,
        np.float64,
        bool,
        npt.NDArray[np.float64],
        int,
        float,
    ]:
        """Return a blended simulation.

        Parameters
        ----------
        idx: int
            The index of the simulation. These refer to the pandas indices
            listed in self.blended_df.
        dx: int
            The amount to dither in the x direction
        dy: int
            The amount to dither in the y direction

        Returns
        -------
        img: np.ndarray, shape=(256, 256)
            image of all blending donuts, centered on the primary donut
        mask: np.ndarray, shape=(256, 256)
            boolean array masking out background and blends
        fx, fy: float
            the field location of the primary donut in radians
        px, py: float
            the focal plane position of the primary donut in radians
        intrafocal: bool
            True = intrafocal, False = extrafocal
        zernikes: np.ndarray, shape=(18,)
            Noll zernikes coefficients 4-21, inclusive
        n_blends: int
            The number of blending stars in this image
        fraction_blended: float
            The fraction of the central donut that is blended
        """

        # get the list of all neighbors
        neighborhood = self.blended_df.loc[idx]
        neighbors = neighborhood.neighborId.tolist()

        # determine the central star
        if self.settings["center_brightest"]:
            center_idx = neighbors.pop(neighborhood.intensity.argmax())
        else:
            center_idx = neighbors.pop(0)

        # get the central star
        (
            img,
            central_mask,
            fx,
            fy,
            px,
            py,
            intra,
            zernikes,
        ) = self._get_unblended(idx, dx, dy, blend_idx=center_idx)

        # keep track of blending stats
        n_blends = 0
        fraction_blended = 0.0

        # now loop over the other neighbors
        neighbor_mask = np.zeros_like(central_mask)
        for n in neighbors:

            # get the neighbor, unblended
            n_img, n_mask, _, _, n_px, n_py, *_ = self._get_unblended(
                idx, dx, dy, blend_idx=n
            )

            # get distance in pixels
            n_dx = round((px - n_px) * self.PIXELS_PER_RADIAN)
            n_dy = round((py - n_py) * self.PIXELS_PER_RADIAN)

            # if we are centering the brightest star, blending stars might
            # be out of frame. We can skip these.
            if (
                abs(n_dx) > 0.9 * n_img.shape[0]
                or abs(n_dy) > 0.9 * n_img.shape[0]
            ):
                continue

            # shift the image and mask
            n_img = self._dither_img(n_img, n_dx, n_dy)
            n_mask = self._dither_img(n_mask, n_dx, n_dy)

            # get the mask for this neighbor
            _neighbor_mask = neighbor_mask + n_mask

            # calculate the fraction blended
            central_overlap = _neighbor_mask[central_mask]
            _fraction_blended = (
                np.count_nonzero(central_overlap) / central_overlap.size
            )

            # if the new fraction blended is too high, skip this star
            if _fraction_blended > self.settings["max_blend"]:
                continue

            # otherwise, update blending stats, and add new blending star
            n_blends += 1
            fraction_blended = _fraction_blended
            img += n_img
            neighbor_mask = _neighbor_mask

        # remove blending regions from the central mask
        mask = np.where(neighbor_mask, 0, central_mask)

        return (
            img,
            mask,
            fx,
            fy,
            px,
            py,
            intra,
            zernikes,
            n_blends,
            fraction_blended,
        )

    def _get_mask(
        self,
        img: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Return mask for central donut.

        Parameters
        ----------
        img: np.ndarray
            Image of the donut

        Returns
        -------
        mask: np.ndarray
            Binary array that masks out background
        """
        threshold = 20
        mask = np.where(img > threshold, True, False)
        if self.settings["mask_buffer"] > 0:
            mask = ndimage.binary_dilation(
                mask, iterations=self.settings["mask_buffer"]
            )
        return mask

    @staticmethod
    def _dither_img(
        img0: npt.NDArray[np.float64], dx: int, dy: int
    ) -> npt.NDArray[np.float64]:
        """Dithers the image.

        Parameters
        ----------
        img0: np.ndarray
            Image to dither
        dx: int
            Number of pixels to shift in the x direction
        dy: int
            Number of pixels to shift in the y direction
        """

        xmax = img0.shape[1]
        ymax = img0.shape[0]

        # select the appropriate slices of the central postage stamp and
        # the dithered postage stamp
        img_xslice = slice(max(0, dx), min(xmax, xmax + dx))
        img_yslice = slice(max(0, dy), min(ymax, ymax + dy))
        img0_xslice = slice(max(0, -dx), min(xmax, xmax - dx))
        img0_yslice = slice(max(0, -dy), min(ymax, ymax - dy))

        # create the central postage stamp of zeros
        img = np.zeros_like(img0)

        # and set the relevant section to the dithered image
        img[img_yslice, img_xslice] = img0[img0_yslice, img0_xslice]

        return img

    def _apply_sky(
        self, img: npt.NDArray[np.float64], seed: int
    ) -> npt.NDArray[np.float64]:
        """Add sky background to the image.

        Parameters
        ----------
        img: np.ndarray
            Image of the donuts.
        seed: int
            Random seed for selecting the sky level.

        Returns
        -------
        np.ndarray
            Image of the donuts plus the sky background.
        """
        # get the sky level
        m_sky = np.random.choice(self.sky)

        # average of ITL + E2V sensors from O’Connor 2019
        gain = (0.69 + 0.94) / 2

        # from https://www.lsst.org/scientists/keynumbers
        plate_scale = 0.2

        # from https://smtn-002.lsst.io/
        m_zero = 28.13

        # exposure time, in seconds
        t_exp = 15

        # average of ITL + E2V sensors from O’Connor 2019
        read_noise = (4.7 + 6.1) / 2

        # generate noise using GalSim
        sky_level = (
            (t_exp / gain) * 10 ** ((m_zero - m_sky) / 2.5) * plate_scale ** 2
        )
        noise = galsim.CCDNoise(
            galsim.BaseDeviate(np.random.randint(2 ** 16)),
            sky_level=sky_level,
            gain=gain,
            read_noise=read_noise,
        )
        galsim_img = galsim.Image(img)
        galsim_img.addNoise(noise)

        return galsim_img.array

    def _apply_badpix(
        self,
        img: npt.NDArray[np.float64],
        mask: npt.NDArray[np.float64],
        seed: int,
    ) -> None:
        """Add bad pixels and columns to the image and mask.

        Parameters
        ----------
        img: np.ndarray
            Image of the donuts.
        mask: np.ndarray
            Mask for the donut.
        seed: int
            Random seed for selecting the bad pixels.
        """

        # select the bad pixels (~2 per image)
        nbadpix = round(np.random.exponential(scale=2))
        x, y = np.random.choice(img.shape[1], nbadpix), np.random.choice(
            img.shape[0], nbadpix
        )
        img[y, x] = 0
        mask[y, x] = 0

        # select the bad columns (~1 per image)
        nbadcol = round(np.random.exponential(scale=1))
        badcols = np.random.choice(img.shape[1], nbadcol, replace=False)
        for col in badcols:
            start = np.random.randint(img.shape[0] - 10)
            end = np.random.randint(start + 1, img.shape[0])
            img[start:end, col] = 0
            mask[start:end, col] = 0

    def _convert_zernikes(
        self, zernikes: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Convert zernike units from r-band wavelength to quadrature
        contribution to FWHM.

        Parameters
        ----------
        zernikes: np.ndarray
            Array of zernike coefficients in units of r-band wavelength

        Returns
        -------
        np.ndarray
            Array of zernike coefficients in units of quadrature contribution
            to FWHM
        """

        rband_eff_wave = 0.6173  # microns

        # these conversion factors depend on telescope radius and obscuration
        # the numbers below are for the Rubin telescope; different numbers
        # are needed for Auxtel. Source: Josh Meyers
        arcsec_per_micron = np.array(
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
