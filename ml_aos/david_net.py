"""Pytorch neural network to predict zernike coefficients from donut images.

My implementation of the network presented in David Thomas's PhD Thesis
at Stanford.
"""
import numpy as np
import torch
from torch import nn


class DavidNet(nn.Module):
    """Network to predict wavefront Zernike coefficients from donut images.

    Consists of a DonutNet that creates image features from the donut image.
    These are concatenated with a set of meta parameters (usually the donut's
    location on the focal plane), which is then passed to the MetaNet, which
    predicts a set of Zernike coefficients.
    """

    def __init__(self, n_meta_layers: int, input_shape: int = 256) -> None:
        """Create a WaveNet to predict Zernike coefficients for donut images.

        Parameters
        ----------
        n_meta_layers: int
            Number of fully connected layers in the MetaNet.
        input_shape: int, default=256
            The shape of the input (square) images. If smaller than 256,
            images will be padded before passed to the network.
        """
        super().__init__()
        self.donut_net = DonutNet()
        self.meta_net = MetaNet(n_meta_layers)

        if (
            (input_shape % 2 != 0)
            or (not isinstance(input_shape, int))
            or (input_shape > 256)
        ):
            raise ValueError("input_shape must be an even integer <= 256")
        elif input_shape == 256:
            self.padder = lambda x: x
        else:
            pad = int((256 - input_shape) / 2)
            self.padder = nn.ZeroPad2d(pad)

    def forward(
        self,
        image: torch.Tensor,
        fx: torch.Tensor,
        fy: torch.Tensor,
        intra: torch.Tensor,
    ) -> torch.Tensor:
        """Predict Zernike coefficients for the donut image.

        Parameters
        ----------
        image: torch.Tensor
            The donut image
        fx: torch.Tensor
            The x angle of the source with respect to the optic axis
        fy: torch.Tensor
            The y angle of the source with respect to the optic axis
        intra: torch.Tensor
            Boolean indicating whether the donut is intra or extra focal

        Returns
        -------
        torch.Tensor
            Array of Zernike coefficients
        """
        padded_image = self.padder(image)
        image_features = self.donut_net(padded_image)
        features = torch.cat([image_features, fx, fy, intra], axis=1)
        return self.meta_net(features)


class DonutNet(nn.Module):
    """Network encodes donut image as latent_dim dimensional latent vector.

    Takes batches of 1x256x256 donut images as input and produces a
    (1 x 1024) dimensional representation.
    """

    def __init__(self) -> None:
        """Create the donut encoder network."""
        super().__init__()

        # first apply a convolution that maintains the image dimensions
        # but increases the channels from 1 to 8
        self.layers = nn.ModuleList(
            [
                nn.Conv2d(1, 8, 3, stride=1, padding=1),
                nn.BatchNorm2d(8),
                nn.ReLU(inplace=True),
            ]
        )

        # now apply a series of DownBlocks that increases the number of
        # channels by a factor of 2, while decreasing height and width
        # by a factor of 2.
        for i in range(7):
            in_channels = 2 ** (i + 3)
            out_channels = 2 ** (i + 3 + 1)
            self.layers.append(DownBlock(in_channels, out_channels))

        # a final down block that doesn't increase the number of channels
        self.layers.append(DownBlock(2 ** 10, 2 ** 10))

        # Finally, flatten the output
        self.layers.append(nn.Flatten())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return latent space encoding of the donut image.

        Parameters
        ----------
        x: torch.Tensor
            Input images of shape (batch x 256 x 256)

        Returns
        -------
        torch.Tensor
            Latent space encoding of shape (batch x 1024)
        """
        for layer in self.layers:
            x = layer(x)
        return x


class DownBlock(nn.Module):
    """Convolutional block that decreases height and width by factor of 2.

    Consists of a convolutional residual/skip layer, followed by a regular
    convolutional layer that decreases the dimensions by a factor of 2.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        """Create a downblock that reduces image dimensions.

        Parameters
        ----------
        in_channels: int
            The number of input channels
        out_channels: int
            The number of output channels
        """
        super().__init__()

        # create the list of layers
        self.layers = nn.ModuleList(
            [
                # residual layer with convolution that preserves dimensions
                SkipBlock(in_channels),
                # this convolution decreases height and width by factor of 2
                nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return a convolved image with half the height and weight.

        Parameters
        ----------
        x: torch.Tensor
            Input image of shape (batch x in_channels x height x width)

        Returns
        -------
        torch.Tensor
            Output image of shape (batch x out_channels x height/2 x width/2)
        """
        for layer in self.layers:
            x = layer(x)
        return x


class SkipBlock(nn.Module):
    """Convolutional layer with a residual/skip connection."""

    def __init__(self, channels: int) -> None:
        """Create a convolution layer with a skip connection.

        Parameters
        ----------
        channels: int
            The number of input and output channels for the convolution.
        """
        super().__init__()

        # layers to compute dx
        self.layers = nn.Sequential(
            nn.Conv2d(channels, channels, 3, stride=1, padding="same"),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Convolve image and add to original via the skip connection.

        Parameters
        ----------
        x: torch.Tensor
            Input image of shape (batch x channels x height x width)

        Returns
        -------
        torch.Tensor
            Output image of shape (batch x channels x height x width)
        """
        dx = self.layers(x)
        return x + dx


class MetaNet(nn.Module):
    """Network that maps image features and meta parameters onto Zernikes.

    Consists of several fully connected layers.
    """

    # number of Zernike coefficients to predict
    N_ZERNIKES = 19
    # number of meta parameters to use in prediction
    N_METAS = 3

    # the dimenson of the image features. This is determined by looking
    # at the dimension of outputs from DonutNet
    IMAGE_DIM = 1024

    def __init__(self, n_layers: int) -> None:
        """Create a MetaNet to map image features and meta params to Zernikes.

        Parameters
        ----------
        n_layers: int
            The number of layers in the MetaNet.
        """
        super().__init__()

        # set number of nodes in network layers using a geometric series
        n_nodes = np.geomspace(
            self.IMAGE_DIM + self.N_METAS,
            self.N_ZERNIKES,
            n_layers + 1,
            dtype=int,
        )

        # create the hidden layers, which all have BatchNorm and ReLU
        self.layers = nn.ModuleList()
        for i in range(n_layers - 1):
            self.layers.extend(
                [
                    nn.Linear(n_nodes[i], n_nodes[i + 1]),
                    nn.BatchNorm1d(n_nodes[i + 1]),
                    nn.ReLU(inplace=True),
                ]
            )

            # we will add dropout to the first layer for regularization
            if i == 0:
                self.layers.append(nn.Dropout(0.1))

        # create the output layer, which doesn't have BatchNorm or ReLU
        self.layers.append(nn.Linear(n_nodes[-2], n_nodes[-1]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map image features and meta parameters onto Zernikes.

        Parameters
        ----------
        x: torch.Tensor
            Input vector of image features and meta parameters.

        Returns
        -------
        torch.Tensor
            Array of Zernike coefficients. Size = cls.N_ZERNIKES

        """
        for layer in self.layers:
            x = layer(x)
        return x
