"""Neural network to predict zernike coefficients from donut images and positions."""

import torch
from torch import nn
from torchvision import models as cnn_models


class WaveNet(nn.Module):
    """Transfer learning driven WaveNet."""

    def __init__(
        self,
        cnn_model: str = "resnet18",
        freeze_cnn: bool = False,
        n_predictor_layers: tuple = (256,),
    ) -> None:
        """Create the WaveNet.

        Parameters
        ----------
        cnn_model: str, default="resnet18"
            The name of the pre-trained CNN model from torchvision.
        freeze_cnn: bool, default=False
            Whether to freeze the CNN weights.
        n_predictor_layers: tuple, default=(256)
            Number of nodes in the hidden layers of the Zernike predictor network.
            This does not include the output layer, which is fixed to 19.
        """
        super().__init__()

        # load the CNN
        if cnn_model != "davidnet":
            self.cnn = getattr(cnn_models, cnn_model)(weights="DEFAULT")

            # save the number of cnn features
            self.n_cnn_features = self.cnn.fc.in_features

            # remove the final fully connected layer
            self.cnn.fc = nn.Identity()
        else:
            raise NotImplementedError("DavidNet not yet implemented.")

        if freeze_cnn:
            # freeze cnn parameters
            self.cnn.eval()
            for param in self.cnn.parameters():
                param.requires_grad = False

        # create linear layers that predict zernikes
        n_meta_features = 4  # includes field_x, field_y, intra flag, wavelen
        n_features = self.n_cnn_features + n_meta_features

        if len(n_predictor_layers) > 0:
            # start with the very first layer
            layers = [
                nn.Linear(n_features, n_predictor_layers[0]),
                nn.BatchNorm1d(n_predictor_layers[0]),
                nn.ReLU(),
            ]

            # add any additional layers
            for i in range(1, len(n_predictor_layers)):
                layers += [
                    nn.Linear(n_predictor_layers[i - 1], n_predictor_layers[i]),
                    nn.BatchNorm1d(n_predictor_layers[i]),
                    nn.ReLU(),
                ]

            # add the final layer
            layers += [nn.Linear(n_predictor_layers[-1], 19)]

        else:
            layers = [nn.Linear(n_features, 19)]

        self.predictor = nn.Sequential(*layers)

    def _reshape_image(self, image: torch.Tensor) -> torch.Tensor:
        """Add 3 identical channels to image tensor."""
        # add a channel dimension
        image = image[..., None, :, :]

        # duplicate image for each channel
        image = image.repeat_interleave(self.cnn.conv1.in_channels, dim=-3)

        return image

    def forward(
        self,
        image: torch.Tensor,
        fx: torch.Tensor,
        fy: torch.Tensor,
        intra: torch.Tensor,
        band: torch.Tensor,
    ) -> torch.Tensor:
        """Predict Zernikes from donut image, location, and wavelength.

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
        band: torch.Tensor
            Float or integer indicating which band the donut was observed in.

        Returns
        -------
        torch.Tensor
            Array of Zernike coefficients (Noll indices 4-23; microns)
        """
        # reshape the image
        image = self._reshape_image(image)

        # use cnn to extract image features
        cnn_features = self.cnn(image)

        # predict zernikes from all features
        features = torch.cat([cnn_features, fx, fy, intra, band], dim=1)
        zernikes = self.predictor(features)

        return zernikes
