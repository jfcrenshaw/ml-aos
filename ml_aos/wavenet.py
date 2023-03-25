"""Neural network to predict zernike coefficients from donut images and positions."""

import torch
from torch import nn
from torchvision import models


class MetaNet(nn.Module):
    """Network to transform the metadata."""

    def __init__(self, n_layers: int = 2, n_nodes: int = 16) -> None:
        """Create the MetaNet.

        Parameters
        ----------
        n_layers : int, default=2
            Number of layers in the MetaNet, including the output layer.
        n_nodes : int, default=16
            Number of nodes in each layer.
        """
        super().__init__()

        # add the very first layer
        layers = [
            nn.Linear(8, n_nodes),
            nn.BatchNorm1d(n_nodes),
        ]

        # add any additional layers
        for _ in range(n_layers - 1):
            layers += [
                nn.ReLU(),
                nn.Linear(n_nodes, n_nodes),
                nn.BatchNorm1d(n_nodes),
            ]

        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pass vector of metadata through the network.

        Parameters
        ----------
        x: torch.Tensor
            Vector of 8 metadata variables.
        """
        return self.layers(x)


class WaveNet(nn.Module):
    """Transfer learning driven WaveNet."""

    def __init__(self, n_meta_layers: int = 2, n_meta_nodes: int = 16) -> None:
        """Create the WaveNet.

        Parameters
        ----------
        n_meta_layers : int, default=2
            Number of layers in the MetaNet, including the output layer.
        n_meta_nodes : int, default=16
            Number of nodes in each layer of the MetaNet.
        """
        super().__init__()

        # first define some parameters that will be accessed by
        # the MachineLearningAlgorithm in ts_wep
        self.camType = "LsstCam"

        # load CNN
        self.cnn = models.resnet18(weights="DEFAULT")

        # save the number of cnn features
        self.n_cnn_features = self.cnn.fc.in_features

        # remove the final fully connected layer
        self.cnn.fc = nn.Identity()

        # freeze cnn parameters
        self.cnn.eval()
        for param in self.cnn.parameters():
            param.requires_grad = False

        # create the MetaNet
        self.meta_net = MetaNet(n_layers=n_meta_layers, n_nodes=n_meta_nodes)

        # create linear layers that predict zernikes
        n_features = self.n_cnn_features + self.meta_net.layers[-1].num_features
        self.predictor = nn.Sequential(
            nn.Linear(n_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 19),
        )

    @staticmethod
    def _reshape_image(image: torch.Tensor) -> torch.Tensor:
        """Add 3 identical channels to image tensor."""
        # add a channel dimension
        image = image[..., None, :, :]

        # duplicate so we have 3 identical channels
        image = image.repeat_interleave(3, dim=-3)

        return image

    def forward(
        self,
        image: torch.Tensor,
        fx: torch.Tensor,
        fy: torch.Tensor,
        intra: torch.Tensor,
        corner: torch.Tensor,
        wavelen: torch.Tensor,
    ) -> torch.Tensor:
        """Predict Zernikes from donut image, location, and wavelength.

        Note most of the inputs are assumed to have been passed through
        ml_aos.utils.transform_inputs.

        Parameters
        ----------
        image: torch.Tensor
            The donut image
        fx: torch.Tensor
            X angle of source with respect to optic axis (radians)
        fy: torch.Tensor
            Y angle of source with respect to optic axis (radians)
        intra: torch.Tensor
            Boolean indicating whether the donut is intra or extra focal
        corner: torch.Tensor
            The one-hot-vector corresponding to the corner where the donut is
            located.
        wavelen: torch.Tensor
            Effective wavelength of the observed band.

        Returns
        -------
        torch.Tensor
            Array of Zernike coefficients (Noll indices 4-23; microns)
        """
        # reshape the image
        image = self._reshape_image(image)

        # use cnn to extract image features
        cnn_features = self.cnn(image)

        # use the MetaNet to extra metadata features
        meta_data = torch.cat([fx, fy, intra, corner, wavelen], dim=1)
        meta_features = self.meta_net(meta_data)

        # predict zernikes from all features
        features = torch.cat([cnn_features, meta_features], dim=1)
        zernikes = self.predictor(features)

        return zernikes
