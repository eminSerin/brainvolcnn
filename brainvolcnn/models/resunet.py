import pytorch_lightning as pl
import torch.nn.functional as F
from torch import nn, optim

from .base_model import BaseModel
from .utils import _BaseLayer, _skip_add, _skip_concat, call_layer


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
        dims=3,
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
        if n_conv is None:
            self.n_conv = 2
        layers = []
        in_ch = self.in_chans
        for i in range(self.n_conv):
            if i == 0 and downsample:
                _stride = 2
            else:
                _stride = self.stride
            layers.append(call_layer("BatchNorm", dims)(in_ch))
            layers.append(self._activation_fn)
            layers.append(
                call_layer("Conv", dims)(
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
            dims=dims,
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

    def __init__(
        self, in_chans, out_chans, stride, kernel_size=3, dims=3, padding=1
    ) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            call_layer("Conv", dims)(
                in_chans,
                out_chans,
                kernel_size=kernel_size,
                padding=padding,
                stride=stride,
            ),
            call_layer("BatchNorm", dims)(out_chans),
        )

    def forward(self, x):
        return self.conv(x)


class _UpSample(pl.LightningModule):
    """Upsampling block."""

    def __init__(
        self, in_chans, out_chans, kernel_size=3, dims=3, stride=2, padding=1
    ) -> None:
        super().__init__()
        self.up = call_layer("ConvTranspose", dims)(
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
        dims=3,
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
            call_layer("Conv", dims)(
                self.in_chans,
                self.out_chans,
                kernel_size=self.kernel_size,
                padding=self.padding,
                stride=2,
            ),
            call_layer("BatchNorm", dims)(self.out_chans),
            self._activation_fn,
            call_layer("Conv", dims)(
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
            dims=dims,
        )

    def forward(self, x):
        skip = self.input_skip(x)
        x = self.input_layer(x)
        return _skip_add(x, skip, mode=self.up_mode)


class _BaseResUNet(BaseModel):
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
        dims=3,
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
                dims=dims,
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
                    dims=dims,
                    padding=self.padding,
                    stride=self.stride,
                    activation=self.activation,
                    up_mode=self.up_mode,
                    downsample=True,
                    n_conv=self.n_conv,
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
                    dims=dims,
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
                    dims=dims,
                )
            )

        self.out_layer = nn.Sequential(
            _UpSample(
                up_features[i + 1],
                out_chans=self.out_chans,
                kernel_size=self.kernel_size,
                dims=dims,
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


class ResUNet1D(_BaseResUNet):
    def __init__(
        self,
        *args,
        dims=1,
        up_mode="linear",
        **kwargs,
    ) -> None:
        super().__init__(
            *args,
            dims=dims,
            up_mode=up_mode,
            **kwargs,
        )


class ResUNet3D(_BaseResUNet):
    def __init__(
        self,
        *args,
        dims=3,
        **kwargs,
    ) -> None:
        super().__init__(
            *args,
            dims=dims,
            **kwargs,
        )
