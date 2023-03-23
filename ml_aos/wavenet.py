"""Neural network to predict zernike coefficients from donut images and positions."""

import torch
from torch import nn
from torchvision import models


class WaveNet(nn.Module):
    """Transfer learning driven WaveNet."""

    def __init__(self) -> None:
        super().__init__()

        # first define some parameters that will be accessed by
        # the MachineLearningAlgorithm in ts_wep
        self.camType = "LsstCam"

        # also define the mean and std for data whitening
        # these were determined from a small sample of the training set
        self.PIXEL_MEAN = 106.82
        self.PIXEL_STD = 154.10

        # load ResNet
        self.resnet = models.resnet18(weights="DEFAULT")

        # remove the final fully connected layer
        self.resnet.fc = nn.Identity()

        # freeze resnet parameters
        self.resnet.eval()
        for param in self.resnet.parameters():
            param.requires_grad = False

        # create the linear layers that predict the zernikes
        self.linear_layers = nn.Sequential(
            nn.Linear(515, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 19),
        )

    @staticmethod
    def _normalize_inputs(
        image: torch.Tensor,
        fx: torch.Tensor,
        fy: torch.Tensor,
        intra: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Normalize the inputs before passing to network."""
        # normalize the image
        PIXEL_MEAN = 106.82
        PIXEL_STD = 154.10
        image = (image - PIXEL_MEAN) / PIXEL_STD

        # normalize the field angles
        FIELD_MEAN = 0
        FIELD_STD = 0.02075
        fx = (fx - FIELD_MEAN) / FIELD_STD
        fy = (fy - FIELD_MEAN) / FIELD_STD

        # normalize the intrafocal flags
        INTRA_MEAN = 0.5
        INTRA_STD = 0.5
        intra = (intra - INTRA_MEAN) / INTRA_STD

        return image, fx, fy, intra

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
    ) -> torch.Tensor:
        """Predict Zernikes from donut image and location .

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

        Returns
        -------
        torch.Tensor
            Array of Zernike coefficients (Noll indices 4-23; microns)
        """
        # normalize the data
        image, fx, fy, intra = self._normalize_inputs(image, fx, fy, intra)

        # reshape the image
        image = self._reshape_image(image)

        # use resnet to extract image features
        image_features = self.resnet(image)

        # predict zernikes from all features
        features = torch.cat([image_features, fx, fy, intra], dim=1)
        zernikes = self.linear_layers(features)

        return zernikes
