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

    # number of blended and unblended simulations
    N_UNBLENDED = 500_000
    N_BLENDED = 100_404

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
        dither: bool = True,
        mask_blends: bool = False,
        fix_seed: bool = True,
        nval: int = 2 ** 16,
        ntest: int = 2 ** 16,
    ):
        """Load the simulated AOS donuts and zernikes in a Pytorch Dataset.

        mode: str, default="train"
            Which set to load. Options are train, val (i.e. validation),
            or test.
        background: bool, default=True
            Whether to add the sky background to the donut images.
        badpix: bool, default=True
            Whether to simulate bad pixels and columns.
        dither: bool, default=True
            Whether to simulate mis-centering by a few pixels.
        mask_blends: bool, default=False
            Whether to mask the blends.
        fix_seed: bool, default=True
            I don't know yet!
        nval: int, default=256
            Number of donuts in the validation set.
        ntest: int, default=2048
            Number of donuts in the test set
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
            "mask_blends": mask_blends,
            "fix_seed": fix_seed,
        }

        # load the sky simulations
        self.sky = np.loadtxt(self.DATA_DIR + "sky.csv")

        # set random seeds
        self._rng_galsim = galsim.BaseDeviate(0)
        self._rng_numpy = np.random.default_rng(0)

        # determine the indices of the val and test sets
        holdout = self._rng_numpy.choice(
            self.N_UNBLENDED + self.N_BLENDED, nval + ntest, replace=False
        )

        # get the indices corresponding to the requested mode
        self.mode = mode
        if mode == "train":
            train = set(np.arange(Donuts.N_UNBLENDED + Donuts.N_BLENDED))
            train -= set(holdout)
            index = np.array(list(train))  # type: npt.NDArray[np.int64]
        elif mode == "val":
            index = holdout[:nval]
        else:
            index = holdout[nval:]

        # load metadata for the unblended images
        unblended_df = pd.read_csv(self.DATA_DIR + "unblended/record.csv")
        unblended_df = unblended_df.iloc[index[index < self.N_UNBLENDED]]
        self.unblended_df = unblended_df.reset_index(drop=True)

        # load metadata for the unblended images
        blended_df = pd.read_csv(self.DATA_DIR + "blended/record.csv")
        blended_df = blended_df.set_index(blended_df["idx"] + self.N_UNBLENDED)
        self.blended_df = blended_df.loc[index[index >= self.N_UNBLENDED]]

    def __len__(self) -> int:
        return len(self.unblended_df) + len(self.blended_df)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Return simulation corresponding to the index."""
        if idx < 0 or idx > len(self):
            raise ValueError("idx out of bounds.")

        # get the simulations for this index
        if idx < self.N_UNBLENDED:
            blend_flag = False
            img, fx, fy, px, py, intra, zernikes = self._get_unblended(idx)
        else:
            blend_flag = True
            img, fx, fy, px, py, intra, zernikes = self._get_blended(idx)

        # apply image distortions
        settings = self.settings
        img = self._apply_sky(idx, img) if settings["background"] else img
        img = self._apply_badpix(idx, img) if settings["badpix"] else img
        img = self._apply_dither(idx, img) if settings["dither"] else img

        # cast NaNs to zero
        img = np.nan_to_num(img)

        # convert outputs to torch tensors and return in a dictionary
        output = {
            "image": torch.from_numpy(img).reshape(1, 256, 256),
            "field_x": torch.FloatTensor([fx]),
            "field_y": torch.FloatTensor([fy]),
            "focal_x": torch.FloatTensor([px]),
            "focal_y": torch.FloatTensor([px]),
            "intrafocal": torch.FloatTensor([intra]),
            "zernikes": torch.from_numpy(zernikes),
            "blended": torch.ByteTensor([blend_flag]),
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
        """Return an unblended simulation."""

        if blend_idx is None:
            img_file = self.DATA_DIR + f"unblended/{idx}.image"
            zernike_file = self.DATA_DIR + f"unblended/{idx}.zernike"
            metadata = self.unblended_df.iloc[idx]
        else:
            file_idx = idx - self.N_UNBLENDED
            img_file = self.DATA_DIR + f"blended/{file_idx}_{blend_idx}.image"
            zernike_file = self.DATA_DIR + f"blended/{file_idx}_0.zernike"
            metadata = self.blended_df.loc[idx].iloc[blend_idx]

        # load the image array
        img = np.load(img_file)

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
    ]:
        """Return a blended simulation."""
        # get the central star
        img, fx, fy, px, py, intra, zernikes = self._get_unblended(
            idx, blend_idx=0
        )

        # now loop over the neighbors
        neighbors = self.blended_df.loc[idx].neighborId[1:]
        for n in neighbors:
            # get the neighbor, unblended
            n_img, _, _, n_px, n_py, *_ = self._get_unblended(idx, blend_idx=n)

            # if we're masking the blends, set all positive pixels to NaN
            # later we will cast all NaNs to zero
            if self.settings["mask_blends"]:
                n_img = np.where(n_img > 10, np.nan, n_img)

            # get distance in pixels
            dx = round((px - n_px) * self.PIXELS_PER_RADIAN)
            dy = round((py - n_py) * self.PIXELS_PER_RADIAN)

            # get slices of central and neighbor images
            img_xslice = slice(max(0, dx), min(256, 256 + dx))
            img_yslice = slice(max(0, dy), min(256, 256 + dy))
            n_img_xslice = slice(max(0, -dx), min(256, 256 - dx))
            n_img_yslice = slice(max(0, -dy), min(256, 256 - dy))

            img[img_yslice, img_xslice] += n_img[n_img_yslice, n_img_xslice]

        return img, fx, fy, px, py, intra, zernikes
