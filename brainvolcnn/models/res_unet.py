import pytorch_lightning as pl
import torch.nn.functional as F
from torch import nn, optim

from .base_model import BaseModel
from .utils import _BaseLayer, _skip_add, _skip_concat


class ResUnit(_BaseLayer):
    """Residual unit.

    Parameters
    ----------
        See `brainvolcnn.models.utils._BaseLayer` for more information.
    """

    def __init__(
        self,
        in_chans,
        out_chans,
        n_conv=2,
        kernel_size=3,
        padding=1,
        stride=1,
        activation="relu_inplace",
        up_mode="trilinear",
        downsample=False,
    ) -> None:
        super().__init__(
            in_chans,
            out_chans,
            n_conv,
            kernel_size,
            padding,
            stride,
            activation,
            up_mode,
        )
        self.downsample = downsample
        layers = []
        in_ch = self.in_chans
        for i in range(self.n_conv):
            if i == 0 and downsample:
                _stride = 2
            else:
                _stride = self.stride
            layers.append(nn.BatchNorm3d(in_ch))
            layers.append(self._activation_fn)
            layers.append(
                nn.Conv3d(
                    in_ch,
                    self.out_chans,
                    kernel_size=self.kernel_size,
                    padding=self.padding,
                    stride=_stride,
                )
            )
            in_ch = self.out_chans
        self.conv = nn.Sequential(*layers)

        if downsample:
            _stride = 2
        else:
            _stride = self.stride
        self.skip_conv = _SkipConv(
            self.in_chans,
            self.out_chans,
            kernel_size=self.kernel_size,
            padding=self.padding,
            stride=_stride,
        )

    def forward(self, x):
        skip = self.skip_conv(x)
        x = self.conv(x)
        return _skip_add(x, skip, mode=self.up_mode)


class _SkipConv(pl.LightningModule):
    """Convolutional block for skip connection.

    Parameters
    ----------
    in_chans : int
        Number of input channels.
    out_chans : int
        Number of output channels.
    stride : int
        Stride of the convolutional layer.
    kernel_size : int
        Kernel size of the convolutional layer.
    padding : int
        Padding of the convolutional layer.
    """

    def __init__(self, in_chans, out_chans, stride, kernel_size=3, padding=1) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(
                in_chans,
                out_chans,
                kernel_size=kernel_size,
                padding=padding,
                stride=stride,
            ),
            nn.BatchNorm3d(out_chans),
        )

    def forward(self, x):
        return self.conv(x)


class _UpSample(pl.LightningModule):
    """Upsampling block."""

    def __init__(self, in_chans, out_chans, kernel_size=3, stride=2, padding=1) -> None:
        super().__init__()
        self.up = nn.ConvTranspose3d(
            in_channels=in_chans,
            out_channels=out_chans,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

    def forward(self, x):
        return self.up(x)


class _InputLayer(_BaseLayer):
    """Input layer."""

    def __init__(
        self,
        in_chans,
        out_chans,
        n_conv=None,
        kernel_size=3,
        padding=1,
        stride=1,
        activation="relu",
        up_mode="trilinear",
    ) -> None:
        super().__init__(
            in_chans,
            out_chans,
            n_conv,
            kernel_size,
            padding,
            stride,
            activation,
            up_mode,
        )
        self.input_layer = nn.Sequential(
            nn.Conv3d(
                self.in_chans,
                self.out_chans,
                kernel_size=self.kernel_size,
                padding=self.padding,
                stride=2,
            ),
            nn.BatchNorm3d(self.out_chans),
            self._activation_fn,
            nn.Conv3d(
                self.out_chans,
                self.out_chans,
                kernel_size=self.kernel_size,
                padding=self.padding,
            ),
        )
        self.input_skip = _SkipConv(
            self.in_chans,
            self.out_chans,
            kernel_size=self.kernel_size,
            padding=self.padding,
            stride=2,
        )

    def forward(self, x):
        skip = self.input_skip(x)
        x = self.input_layer(x)
        return _skip_add(x, skip, mode=self.up_mode)


class ResUNet(BaseModel):
    """ResUNet architecture.

    Deep-learning based segmentation model for 3D medical images, combining
    deep residual networks and U-net architecture [1].

    Parameters
    ----------
    See models.base_model.BaseModel for more details.


    References
    ----------
    [1] - Zhang, Zhengxin, Qingjie Liu, and Yunhong Wang. "Road extraction by deep  residual u-net." IEEE Geoscience and Remote Sensing Letters 15.5 (2018): 749-753.
    """

    def __init__(
        self,
        in_chans,
        out_chans,
        max_level=5,
        fdim=64,
        n_conv=2,
        kernel_size=3,
        padding=1,
        stride=1,
        activation="relu",
        up_mode="trilinear",
        final_activation=None,
        loss_fn=F.mse_loss,
        optimizer=optim.Adam,
        lr=0.001,
        **kwargs,
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
            **kwargs,
        )
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.upscale = nn.ModuleList()

        # Downsampling including Bridge
        self.downs.append(
            _InputLayer(
                in_chans,
                self._features[0],
                kernel_size=kernel_size,
                padding=self.padding,
                stride=self.stride,
                activation=self.activation,
                up_mode=self.up_mode,
            )
        )
        for i in range(1, max_level):
            self.downs.append(
                ResUnit(
                    self._features[i - 1],
                    self._features[i],
                    kernel_size=kernel_size,
                    padding=self.padding,
                    stride=self.stride,
                    activation=self.activation,
                    up_mode=self.up_mode,
                    downsample=True,
                )
            )

        # Upsampling
        up_features = self._features[::-1]
        for i in range(max_level - 1):
            self.ups.append(
                ResUnit(
                    up_features[i] + up_features[i + 1],
                    up_features[i + 1],
                    n_conv=self.n_conv,
                    kernel_size=self.kernel_size,
                    padding=self.padding,
                    stride=self.stride,
                    activation=self.activation,
                    up_mode=self.up_mode,
                )
            )

            self.upscale.append(
                _UpSample(
                    up_features[i],
                    up_features[i],
                    kernel_size=self.kernel_size,
                    padding=self.padding,
                )
            )

        self.out_layer = nn.Sequential(
            _UpSample(
                up_features[i + 1],
                out_chans=self.out_chans,
                kernel_size=self.kernel_size,
                padding=self.padding,
                stride=2,
            ),
        )

    def forward(self, x):
        # Downward path
        skip_connections = []
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
        skip_connections.pop(-1)
        skip_connections = skip_connections[::-1]

        # Upward path
        for up, skip, scale in zip(self.ups, skip_connections, self.upscale):
            # x = _UpSample(x.shape[1], x.shape[1])(x)
            x = scale(x)
            x = _skip_concat(x, skip, mode=self.up_mode)
            x = up(x)

        return self.out_layer(x)
