"""_summary_
"""
import torch.nn.functional as F
from torch import nn, optim

from .base_model import BaseModel
from .utils import _nConv, _skip_concat


class UNet(BaseModel):
    """Implementation of U-Net: Convolutional Networks for Biomedical Image Segmentation[1].

    ![U-Net Architecture](./_imgs/u-net.png)

    Attributes
    ----------
    in_channel : int
        Number of input channels.
    out_channel : int
        Number of output channels.
    max_level : int, optional
        Maximum level of downstream and upstreams, by default 5
    fdim : int, optional
        Initial number of features, by default 64.
        In each level, the number of features is doubled.
    up_mode : str, optional
        Upscaling mode, by default "nearest-exact".
        Upscaling is used if the input and output shapes are different.
    **args, **kwargs: dict
        Arguments for the BaseModel class.
        See brainvolcnn.models.base_model.BaseModel for more details.

    References
    ----------
    [1] - Ronneberger, Olaf, Philipp Fischer, and Thomas Brox. "U-net: Convolutional networks for biomedical image segmentation." International Conference on Medical image computing and computer-assisted intervention. Springer, Cham, 2015.
    """

    def __init__(
        self,
        in_chans,
        out_chans,
        max_level=4,
        fdim=64,
        n_conv=3,
        kernel_size=3,
        padding=1,
        stride=1,
        activation="relu_inplace",
        up_mode="trilinear",
        final_activation="relu_inplace",
        loss_fn=F.mse_loss,
        optimizer=optim.Adam,
        lr=0.001,
    ) -> None:
        super().__init__(
            in_chans,
            out_chans,
            max_level,
            fdim,
            n_conv,
            kernel_size,
            padding,
            stride,
            activation,
            up_mode,
            final_activation,
            loss_fn,
            optimizer,
            lr,
        )
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

        # Down
        in_dim = self.in_chans
        for feat in self._features:
            self.downs.append(_nConv(in_dim, feat, n_conv=2))
            in_dim = feat

        self.bottleneck = _nConv(
            feat, feat * 2, n_conv=2, kernel_size=self.kernel_size, padding=self.padding
        )

        # Up
        for feat in reversed(self._features):
            self.ups.append(nn.ConvTranspose3d(feat * 2, feat, kernel_size=2, stride=2))
            self.ups.append(
                _nConv(
                    feat * 2,
                    feat,
                    n_conv=2,
                    kernel_size=self.kernel_size,
                    padding=self.padding,
                )
            )

        self.final_conv = nn.Sequential(
            nn.Conv3d(feat, self.out_chans, kernel_size=1), self.final_activation
        )

    def forward(self, x):
        skip_connections = []
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            sc = skip_connections[idx // 2]
            x = self.ups[idx + 1](_skip_concat(x, sc, mode=self.up_mode))
        return self.final_conv(x)
