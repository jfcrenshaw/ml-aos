"""Pytorch DataSet for the AOS simulations."""
import glob
from typing import Any, Dict

import numpy as np
import torch
from astropy.table import Table
from torch.utils.data import Dataset

from ml_aos.utils import transform_inputs


class Donuts(Dataset):
    """AOS donuts and zernikes from my simulations."""

    def __init__(
        self,
        mode: str = "train",
        transform: bool = True,
        data_dir: str = "/astro/store/epyc/users/jfc20/data/aos_sims",
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
        data_dir: str, default=/astro/store/epyc/users/jfc20/aos_sims
            Location of the data directory. The default location is where
            I stored the simulations on epyc.
        """
        # save the settings
        self.settings = {
            "mode": mode,
            "transform": transform,
            "data_dir": data_dir,
        }

        # get a list of all the observations
        all_image_files = glob.glob(f"{data_dir}/images/*")
        obs_ids = list(
            set(
                [int(file.split("/")[-1].split(".")[1][3:]) for file in all_image_files]
            )
        )

        # get the table of metadata for each observation
        observations = Table.read(f"{data_dir}/opSimTable.parquet")
        observations = observations[obs_ids]
        self.observations = observations

        # now split the observations between train, test, val
        train_ids = []
        val_ids = []
        test_ids = []

        # we don't have enough u band, so let's put 2 in test and rest in train
        group = observations[observations["lsstFilter"] == "u"]
        test_ids.extend(group["observationId"][:2])
        train_ids.extend(group["observationId"][2:])

        # for the rest of the bands, let's put 2 each in test/val, and rest in train
        for band in "grizy":
            group = observations[observations["lsstFilter"] == band]
            test_ids.extend(group["observationId"][:2])
            val_ids.extend(group["observationId"][2:4])
            train_ids.extend(group["observationId"][4:])

        self.obs_ids = {
            "train": train_ids,
            "val": val_ids,
            "test": test_ids,
        }

        # partition the image files
        self.image_files = {
            mode: [
                file
                for file in all_image_files
                if int(file.split("/")[-1].split(".")[1][3:]) in ids
            ]
            for mode, ids in self.obs_ids.items()
        }

    def __len__(self) -> int:
        """Return length of this Dataset."""
        return len(self.image_files[self.settings["mode"]])  # type: ignore

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
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
        img_file = self.image_files[self.settings["mode"]][idx]  # type: ignore

        # load the image
        img = np.load(img_file)

        # crop out the central 160x160
        img = img[5:-5, 5:-5]

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
        band = "ugrizy".index(obs_row["lsstFilter"].item())

        # load the zernikes
        zernikes = np.load(
            (
                f"{self.settings['data_dir']}/zernikes/"
                f"{pntId}.{obsId}.detector{row['detector'][:3]}.zernikes.npy"
            )
        )

        # load the degrees of freedom
        dof = np.load(f"{self.settings['data_dir']}/dof/{pntId}.dofs.npy")

        # standardize all the inputs for the neural net
        if self.settings["transform"]:
            img, fx, fy, intra, band = transform_inputs(
                img,
                fx,
                fy,
                intra,
                band,
            )

        # convert everything to tensors
        img = torch.from_numpy(img).float()
        fx = torch.FloatTensor([fx])
        fy = torch.FloatTensor([fy])
        intra = torch.FloatTensor([intra])
        band = torch.FloatTensor([band])
        zernikes = torch.from_numpy(zernikes).float()
        dof = torch.from_numpy(dof).float()

        output = {
            "image": img,
            "field_x": fx,
            "field_y": fy,
            "intrafocal": intra,
            "band": band,
            "zernikes": zernikes,
            "dof": dof,
            "pntId": int(pntId[3:]),
            "obsId": int(obsId[3:]),
            "objId": int(objId[3:]),
        }

        return output
