"""Pytorch DataSet for the AOS simulations.

Based on code written by David Thomas for his PhD Thesis at Stanford.
"""
import glob
from typing import Any

import numpy as np
import torch
from astropy.table import Table
from torch.utils.data import Dataset

from ml_aos.utils import convert_zernikes


class Donuts(Dataset):
    """AOS donuts and zernikes from my simulations."""

    def __init__(
        self,
        mode: str = "train",
        convert_zernikes: bool = False,
        nval: int = 2048,
        ntest: int = 2048,
        data_dir: str = "/astro/store/epyc/users/jfc20/aos_sims",
        **kwargs: Any,
    ) -> None:
        """Load the simulated AOS donuts and zernikes in a Pytorch Dataset.

        Parameters
        ----------
        mode: str, default="train"
            Which set to load. Options are train, val (i.e. validation),
            or test.
        convert_zernikes: bool, default=False
            Whether to convert Zernike coefficients from units of r band
            wavelength to quadrature contribution to PSF FWHM.
        nval: int, default=256
            Number of donuts in the validation set.
        ntest: int, default=2048
            Number of donuts in the test set
        data_dir: str, default=/astro/store/epyc/users/jfc20/aos_sims
            Location of the data directory. The default location is where
            I stored the simulations on epyc.
        """
        # save the settings
        self.settings = {
            "convert_zernikes": convert_zernikes,
            "data_dir": data_dir,
        }

        # get the image files
        image_files = glob.glob(f"{data_dir}/images/*")

        # get the test set
        test_set = image_files[-ntest:]
        testIds = list(set([file.split("/")[-1].split(".")[0] for file in test_set]))

        # remove the non-test set images that were in the same observation as a test
        # set image (because they have the same perturbations)
        rest = [
            file
            for file in image_files
            if not any(testId in file for testId in testIds)
        ]

        # get the validation set
        val_set = rest[-nval:]
        valIds = list(set([file.split("/")[-1].split(".")[0] for file in val_set]))

        # remove the non-validation set images that were in the same observation as a
        # validation set image (because they have the same perturbations)
        rest = [file for file in rest if not any(valId in file for valId in valIds)]

        # the rest of the files will be used for training
        train_set = rest

        # set the image files to the requested set
        if mode == "train":
            self._image_files = train_set
        elif mode == "val":
            self._image_files = val_set
        elif mode == "test":
            self._image_files = test_set

        # get the table of metadata for each observation
        self.observations = Table.read(f"{data_dir}/opSimTable.parquet")

    def __len__(self) -> int:
        """Return length of this Dataset."""
        return len(self._image_files)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Return simulation corresponding to the index.

        Parameters
        ----------
        idx: int
            The index of the simulation to return.

        Returns
        -------
        dict
            The dictionary contains the following pytorch tensors
                image: donut image, shape=(256, 256)
                mask: donut mask, shape=(256, 256)
                field_x, field_y: the field angle in radians
                detector_x, detector_y: position on the detector in pixels
                intrafocal: boolean flag. 0 = extrafocal, 1 = intrafocal
                zernikes: Noll zernikes coefficients 4-21, inclusive (microns)
                n_blends: the number of blending stars in the image
                fraction_blended: fraction of the central donut blended
                pointing: the pointing ID
        """
        # get the image file
        img_file = self._image_files[idx]

        # load the image
        img = np.load(img_file)

        # get the IDs
        pntId, obsId, objId = img_file.split("/")[-1].split(".")[:3]

        # get the catalog for this observation
        catalog = Table.read(
            f"{self.settings['data_dir']}/catalogs/{pntId}.catalog.parquet"
        )
        self.catalog = catalog

        # get the row for this source
        row = catalog[catalog["objectId"] == int(objId[3:])][0]

        # get the donut locations
        fx, fy = row["xField"], row["yField"]

        # get the intra/extra flag
        intra = "SW1" in row["detector"]

        # load the zernikes
        zernikes = np.load(
            (
                f"{self.settings['data_dir']}/zernikes/"
                f"{pntId}.{obsId}.detector{row['detector'][:3]}.zernikes.npy"
            )
        )

        # convert zernikes to their contribution to FWHM
        if self.settings["convert_zernikes"]:
            zernikes = convert_zernikes(zernikes)

        # load the degrees of freedom
        dof = np.load(f"{self.settings['data_dir']}/dof/{pntId}.dofs.npy")

        pntId, obsId, objId
        output = {
            "image": torch.from_numpy(img).float(),
            "field_x": torch.FloatTensor([fx]),
            "field_y": torch.FloatTensor([fy]),
            "intrafocal": torch.FloatTensor([intra]),
            "zernikes": torch.from_numpy(zernikes).float(),
            "dof": torch.from_numpy(dof).float(),
            "pntId": int(pntId[3:]),
            "obsId": int(obsId[3:]),
            "objId": int(objId[3:]),
        }

        return output
