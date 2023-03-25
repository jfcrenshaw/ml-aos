"""Pytorch DataSet for the AOS simulations.

Based on code written by David Thomas for his PhD Thesis at Stanford.
"""
import glob
from typing import Any

import numpy as np
import torch
from astropy.table import Table
from torch.utils.data import Dataset

from ml_aos.utils import get_corner, transform_inputs


class Donuts(Dataset):
    """AOS donuts and zernikes from my simulations."""

    def __init__(
        self,
        mode: str = "train",
        transform: bool = True,
        nval: int = 2**12,
        ntest: int = 2**12,
        data_dir: str = "/astro/store/epyc/users/jfc20/aos_sims",
        seed: int = 0,
        **kwargs: Any,
    ) -> None:
        """Load the simulated AOS donuts and zernikes in a Pytorch Dataset.

        Parameters
        ----------
        mode: str, default="train"
            Which set to load. Options are train, val (i.e. validation),
            or test.
        transform: bool, default=True
            Whether to apply transform_inputs from ml_aos.utils.
        nval: int, default=256
            Number of donuts in the validation set.
        ntest: int, default=2048
            Number of donuts in the test set
        data_dir: str, default=/astro/store/epyc/users/jfc20/aos_sims
            Location of the data directory. The default location is where
            I stored the simulations on epyc.
        seed: int, default = 0
            Random seed for shuffling the data files.
        """
        # save the settings
        self.settings = {
            "transform": transform,
            "data_dir": data_dir,
        }

        # get the image files
        image_files = glob.glob(f"{data_dir}/images/*")

        # shuffle the image files
        # rng = np.random.default_rng(seed)
        # rng.shuffle(image_files)

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
                field_x, field_y: the field angle in radians
                intrafocal: boolean flag. 0 = extrafocal, 1 = intrafocal
                band: LSST band indicated by index in string "ugrizy" (e.g. 2 = "r")
                zernikes: Noll zernikes coefficients 4-21, inclusive (microns)
                dof: the telescope perturbations corresponding to the zernikes
                pntId: the pointing ID
                obsID: the observation ID
                objID: the object ID
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

        # get the row for this source
        row = catalog[catalog["objectId"] == int(objId[3:])][0]

        # get the donut locations
        fx, fy = row["xField"], row["yField"]

        # get the intra/extra flag
        intra = "SW1" in row["detector"]

        # get the observed band
        obs_row = self.observations[
            self.observations["observationId"] == int(obsId[3:])
        ]
        band = obs_row["lsstFilter"].item()

        # get the effective wavelength in microns
        wavelen = {
            "u": 0.3671,
            "g": 0.4827,
            "r": 0.6223,
            "i": 0.7546,
            "z": 0.8691,
            "y": 0.9712,
        }[band]

        # load the zernikes
        zernikes = np.load(
            (
                f"{self.settings['data_dir']}/zernikes/"
                f"{pntId}.{obsId}.detector{row['detector'][:3]}.zernikes.npy"
            )
        )

        # load the degrees of freedom
        dof = np.load(f"{self.settings['data_dir']}/dof/{pntId}.dofs.npy")

        # convert everything to tensors
        img = torch.from_numpy(img).float()
        fx = torch.FloatTensor([fx])
        fy = torch.FloatTensor([fy])
        intra = torch.FloatTensor([intra])
        wavelen = torch.FloatTensor([wavelen])
        zernikes = torch.from_numpy(zernikes).float()
        dof = torch.from_numpy(dof).float()

        # standardize all the inputs for the neural net
        if self.settings["transform"]:
            img, fx, fy, intra, wavelen, corner = transform_inputs(
                img,
                fx,
                fy,
                intra,
                wavelen,
            )
        else:
            corner = torch.FloatTensor([get_corner(fx, fy)])

        output = {
            "image": img,
            "field_x": fx,
            "field_y": fy,
            "corner": corner,
            "intrafocal": intra,
            "wavelen": wavelen,
            "zernikes": zernikes,
            "dof": dof,
            "band": band,
            "pntId": int(pntId[3:]),
            "obsId": int(obsId[3:]),
            "objId": int(objId[3:]),
        }

        return output
