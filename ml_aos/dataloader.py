"""Pytorch DataSet for the AOS simulations.

Based on code written by David Thomas for his PhD Thesis at Stanford.
"""

import galsim
import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
from torch.utils.data import Dataset


class Donuts(Dataset):
    """Data set of AOS donuts and zernikes."""

    # location of the simulations on epyc
    DATA_DIR = "/epyc/users/jfc20/thomas_aos_sims/"

    # number of blended and unblended simulations in the full set
    _N_UNBLENDED = 500_000
    _N_BLENDED = 100_404

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
        mask_blends: bool = False,
        center_brightest: bool = True,
        nval: int = 2 ** 16,
        ntest: int = 2 ** 16,
        seed: int = 0,
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
        mask_blends: bool, default=False
            Whether to mask the blends.
        center_brightest: bool, default=True
            Whether to center the brightest star in blended images.
        nval: int, default=256
            Number of donuts in the validation set.
        ntest: int, default=2048
            Number of donuts in the test set
        seed: int, default=0
            Random seed for training set/test set/validation set selection.
        """
        # check that the mode is valid
        allowed_modes = ["train", "val", "test"]
        if mode not in allowed_modes:
            raise ValueError(
                f"mode must be one of {', '.join(allowed_modes)}."
            )

        # set the image properties
        self.settings = {
            "background": background,
            "badpix": badpix,
            "dither": dither,
            "max_blend": max_blend,
            "mask_blends": mask_blends,
            "center_brightest": center_brightest,
        }

        # determine the indices of the val and test sets
        rng = np.random.default_rng(seed)
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
        return len(self.unblended_df) + len(self.blended_df)

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
                image: cenered donut image, shape=(256, 256)
                field_x, field_y: the field location in radians
                focal_x, focal_y: the focal plane position in radians
                intrafocal: boolean flag. 0 = extrafocal, 1 = intrafocal
                zernikes: Noll zernikes coefficients 4-21, inclusive
                n_blends: the number of blending stars in the image
                fraction_blended: fraction of the central donut blended
        """
        if idx < 0 or idx > len(self):
            raise ValueError("idx out of bounds.")

        # get the simulations for this index
        if idx < self.N_unblended:
            img, fx, fy, px, py, intra, zernikes = self._get_unblended(idx)
            nb = 0  # n_blends=0
            fb = 0.0  # fraction_blended=0
        else:
            img, fx, fy, px, py, intra, zernikes, nb, fb = self._get_blended(
                idx
            )

        # sky background and bad pixels (if requested)
        settings = self.settings
        img = self._apply_sky(img, seed=idx) if settings["background"] else img
        img = self._apply_badpix(img, seed=idx) if settings["badpix"] else img

        # cast NaNs to zero
        img = np.nan_to_num(img)

        # convert outputs to torch tensors and return in a dictionary
        output = {
            "image": torch.from_numpy(img).reshape(1, 256, 256),
            "field_x": torch.FloatTensor([fx]),
            "field_y": torch.FloatTensor([fy]),
            "focal_x": torch.FloatTensor([px]),
            "focal_y": torch.FloatTensor([py]),
            "intrafocal": torch.FloatTensor([intra]),
            "zernikes": torch.from_numpy(zernikes),
            "n_blends": torch.ByteTensor([nb]),
            "fraction_blended": torch.FloatTensor([fb]),
        }

        return output

    def _get_unblended(
        self,
        idx: int,
        blend_idx: int = None,
    ) -> tuple[
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
        blend_idx: int, optional
            The neighborId lists in self.blended_df.

        Returns
        -------
        img: np.ndarray, shape=(256, 256)
            centered donut image
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
        img0 = np.load(img_file)

        # apply dither
        # randomly select dither size
        rng = np.random.default_rng(seed=idx)
        dither = self.settings["dither"]
        dx, dy = rng.integers(-dither, dither + 1, size=2)
        # select the appropriate slices of the central postage stamp and
        # the dithered postage stamp
        img_xslice = slice(max(0, dx), min(256, 256 + dx))
        img_yslice = slice(max(0, dy), min(256, 256 + dy))
        img0_xslice = slice(max(0, -dx), min(256, 256 - dx))
        img0_yslice = slice(max(0, -dy), min(256, 256 - dy))
        # create the central postage stamp of zeros
        img = np.zeros_like(img0)
        # and set the relevant section to the dithered image
        img[img_yslice, img_xslice] = img0[img0_yslice, img0_xslice]

        # load the field position, in radians
        fx, fy = metadata.fieldx, metadata.fieldy

        # load the focal plane position, in radians
        px, py = metadata.posx, metadata.posy

        # boolean stating whether the image is intrafocal
        # obviously, if false, the image is extrafocal
        intra = "SW0" in metadata.chip

        # load the zernike coefficients; we use 4-21, inclusive
        zernikes = np.load(zernike_file)[4:22]

        return img, fx, fy, px, py, intra, zernikes

    def _get_blended(
        self,
        idx: int,
    ) -> tuple[
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

        Returns
        -------
        img: np.ndarray, shape=(256, 256)
            image of all blending donuts, centered on the primary donut
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
        img, fx, fy, px, py, intra, zernikes = self._get_unblended(
            idx, blend_idx=center_idx
        )

        # get a mask of the central star to determine fraction blended
        mask_cut = 10
        central_mask = np.where(img > mask_cut, True, False)

        # keep track of blending stats
        n_blends = 0
        fraction_blended = 0.0

        # now loop over the other neighbors
        neighbor_mask = np.zeros_like(img)
        for n in neighbors:

            # get the neighbor, unblended
            n_img, _, _, n_px, n_py, *_ = self._get_unblended(idx, blend_idx=n)

            # get distance in pixels
            dx = round((px - n_px) * self.PIXELS_PER_RADIAN)
            dy = round((py - n_py) * self.PIXELS_PER_RADIAN)

            # get slices of central and neighbor images
            img_xslice = slice(max(0, dx), min(256, 256 + dx))
            img_yslice = slice(max(0, dy), min(256, 256 + dy))
            n_img_xslice = slice(max(0, -dx), min(256, 256 - dx))
            n_img_yslice = slice(max(0, -dy), min(256, 256 - dy))

            # shift the neighbor image
            sn_img = np.zeros_like(n_img)
            sn_img[img_yslice, img_xslice] += n_img[n_img_yslice, n_img_xslice]

            # get the mask for this neighbor
            n_mask = np.where(sn_img > mask_cut, True, False)
            _neighbor_mask = neighbor_mask + n_mask

            # calculate the fraction blended
            central_overlap = _neighbor_mask[central_mask]
            _fraction_blended = (
                np.count_nonzero(central_overlap) / central_overlap.size
            )

            # if the new fraction blended is too high, don't add more stars!
            if _fraction_blended > self.settings["max_blend"]:
                break

            # otherwise, update blending stats, and add new blending star
            n_blends += 1
            fraction_blended = _fraction_blended
            img += sn_img
            neighbor_mask = _neighbor_mask

        # if masking blends, set blending stars to NaN. We will cast NaNs
        # to zero at the end. We do this so that if you add sky background,
        # you don't have sky background in the masked regions.
        if self.settings["mask_blends"]:
            img += np.where(neighbor_mask, np.nan, 0)

        return img, fx, fy, px, py, intra, zernikes, n_blends, fraction_blended

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
        rng = np.random.default_rng(seed)
        m_sky = rng.choice(self.sky)

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
            galsim.BaseDeviate(seed),
            sky_level=sky_level,
            gain=gain,
            read_noise=read_noise,
        )
        galsim_img = galsim.Image(img)
        galsim_img.addNoise(noise)

        return galsim_img.array

    def _apply_dither(
        self, img: npt.NDArray[np.float64], seed: int
    ) -> npt.NDArray[np.float64]:
        """Dither the image to simulate miscentering.

        Parameters
        ----------
        img: np.ndarray
            Image of the donuts.
        seed: int
            Random seed for selecting the dither.

        Returns
        -------
        np.ndarray
            Image of the donuts plus the sky background.
        """
        rng = np.random.default_rng(seed)

        # randomly determine dither, [-5, 5] in each dimension
        dx, dy = rng.integers(-5, 6, size=2)

        # apply dither
        img = np.roll(np.roll(img, dx, 1), dy, 0)

        return img

    def _apply_badpix(
        self, img: npt.NDArray[np.float64], seed: int
    ) -> npt.NDArray[np.float64]:
        """Add bad pixels and columns to the image.

        Parameters
        ----------
        img: np.ndarray
            Image of the donuts.
        seed: int
            Random seed for selecting the bad pixels.

        Returns
        -------
        np.ndarray
            Image of the donuts including bad pixels.
        """
        return_img = img.copy()

        # select the bad pixels (~2 per image)
        rng = np.random.default_rng(seed)
        nbadpix = round(rng.exponential(scale=2))
        x, y = rng.choice(256, nbadpix), rng.choice(256, nbadpix)
        return_img[x, y] = 0

        # select the bad columns (~1 per image)
        nbadcol = round(rng.exponential(scale=1))
        badcols = rng.choice(256, nbadcol, replace=False)
        for col in badcols:
            start = rng.integers(256)
            end = rng.integers(start + 1, 256)
            return_img[start:end, col] = 0

        return return_img
