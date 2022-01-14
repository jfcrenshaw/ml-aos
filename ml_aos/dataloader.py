"""Pytorch DataSet for the AOS simulations."""
import galsim
import numpy as np
import numpy.typing as npt
import pandas as pd
from torch.utils.data import Dataset


class Donuts(Dataset):
    """Data set of AOS donuts and zernikes."""

    # location of the simulations on epyc
    data_dir = "/epyc/users/jfc20/thomas_aos_sims/"

    # number of blended and unblended simulations
    N_UNBLENDED = 500_000
    N_BLENDED = 100_404

    def __init__(
        self,
        mode: str = "train",
        background: bool = True,
        badpix: bool = True,
        dither: bool = True,
        mask_blends: bool = False,
        fixseed: bool = True,
        nval: int = 256,
        ntest: int = 2048,
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
        fixseed: bool, default=True
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
        self.background = background
        self.badpix = badpix
        self.dither = dither
        self.mask_blends = mask_blends
        self.fixseed = fixseed

        # load the metadata for the unblended and blended images
        self.unblended_df = pd.read_csv(self.data_dir + "unblended/record.csv")
        self.blended_df = pd.read_csv(self.data_dir + "blended/record.csv")

        # load the sky simulations
        self.sky = np.loadtxt(self.data_dir + "sky.csv")

        # set random seeds
        self.galsim_rng = galsim.BaseDeviate(0)
        self.numpy_rng = np.random.default_rng(0)

        # determine the indices of the val and test sets
        holdout = self.numpy_rng.choice(
            self.N_UNBLENDED + self.N_BLENDED, nval + ntest, replace=False
        )

        # save the index corresponding to the requested mode
        self.mode = mode
        if mode == "train":
            self.len = Donuts.N_UNBLENDED + Donuts.N_BLENDED - nval - ntest
            train = set(np.arange(self.len)) - set(holdout)
            self.index = np.array(list(train))  # type: npt.NDArray[np.int64]
        elif mode == "val":
            self.len = nval
            self.index = holdout[:nval]
        else:
            self.len = ntest
            self.index = holdout[nval:]
