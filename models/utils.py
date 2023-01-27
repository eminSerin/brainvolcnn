"""_summary_

Returns
-------
_type_
    _description_

Raises
------
NotImplementedError
    _description_
"""

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F


class _BaseLayer(pl.LightningModule):
    """Base layer object for all layers used in
    the U-Net and V-Net architectures.

    Arguments:
    ----------
    in_chans : int
        Number of input channels.
    out_chans : int
        Number of output channels.
    n_conv : int
        Number of convolutional layers, by default None.
    kernel_size : int
        Convolutional kernel size, by default 3.
    padding : int
        Convolutional padding, by default 1.
    stride : int
        Convolutional stride, by default 1.
    activation : str, optional
        Activation function.
        See `brainvolcnn.models.utils._activation_fn`
        for more details, by default "relu".
    up_mode : str, optional
        Upscaling mode, by default "trilinear".
        Upscaling is used if the input and target
        shapes are different.

    """

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
        super().__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.n_conv = n_conv
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.activation = activation
        self._activation_fn = _activation_fn(activation, out_chans)
        self.up_mode = up_mode


class _nConv(_BaseLayer):
    """Performs n numbers of convolution
    required in hourglass neural
    network architectures.
    """

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
        layers = []
        in_ch = self.in_chans
        for _ in range(n_conv):
            layers.append(
                nn.Conv3d(
                    in_ch,
                    self.out_chans,
                    kernel_size=self.kernel_size,
                    padding=self.padding,
                    stride=self.stride,
                )
            )
            layers.append(nn.BatchNorm3d(self.out_chans))
            layers.append(self._activation_fn)
            in_ch = self.out_chans

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv(x)


def _skip_concat(x, skip, mode="trilinear"):
    """Concatenates the skip connection

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.
    skip : torch.Tensor
        Skip connection tensor.
    mode : str, optional
        Interpolation method, by default "trilinear"
        See `torch.nn.functional.interpolate` for more
        information.

    Returns
    -------
    torch.Tensor
        Output tensor.
    """
    if x.shape != skip.shape:
        x = _interpolate(x, skip.shape[2:], mode=mode)
    return torch.cat([x, skip], dim=1)


def _skip_add(x, skip, mode="trilinear"):
    """Addes the skip connection

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.
    skip : torch.Tensor
        Skip connection tensor.
    mode : str, optional
        Interpolation method, by default "trilinear"
        See `torch.nn.functional.interpolate` for more
        information.

    Returns
    -------
    torch.Tensor
        Output tensor.
    """
    if x.shape != skip.shape:
        x = _interpolate(x, skip.shape[2:], mode=mode)
    return torch.add(x, skip)


def _interpolate(X, size, mode="trilinear", align_corners=None):
    """_summary_

    Parameters
    ----------
    X : torch.Tensor
        Input tensor.
    size : tuple
        Tuple of target size.
    mode : str, optional
        Interpolation method, by default "trilinear".
        See torch.nn.functional.interpolate for more details.
    align_corners : bool or None, optional
        Whether to align corners, by default None.
        It only works for mode "bilinear" and "trilinear".

    Returns
    -------
    torch.Tensor
        Interpolated tensor.
    """
    if mode not in ["nearest", "nearest-exact"]:
        align_corners = None
    else:
        align_corners = True
    return F.interpolate(X, size=size, mode=mode, align_corners=align_corners)


def _activation_fn(activation="relu_inplace", n_channels=None):
    """It returns the activation function.

    Parameters
    ----------
    activation : str
        Type of activation function.
        It currently supports "relu", "leakyrelu",
        "prelu", "elu", "tanh", "sigmoid",
        by default "relu". For ReLU, LeakyReLU,
        and ELU, it returns the inplace version
        if you add "_inplace" to the activation
        function name. For example, "relu_inplace".
    n_channels : int, optional
        Number of parameters for PReLU activation
        function, by default None. This is only
        for PReLU activation function. If None,
        it is set to 1.
    Returns
    -------
    nn.Module
        Activation function.
    """
    if activation == "relu":
        return nn.ReLU(inplace=False)
    elif activation == "relu_inplace":
        return nn.ReLU(inplace=True)
    elif activation == "leakyrelu":
        return nn.LeakyReLU(inplace=False)
    elif activation == "leakyrelu_inplace":
        activation = nn.LeakyReLU(inplace=True)
    elif activation == "elu":
        return nn.ELU(inplace=False)
    elif activation == "elu_inplace":
        return nn.ELU(inplace=True)
    elif activation == "tanh":
        return nn.Tanh()
    elif activation == "sigmoid":
        return nn.Sigmoid()
    elif activation == "prelu":
        if n_channels is None:
            n_channels = 1
        return nn.PReLU(num_parameters=n_channels, init=0.25)
    else:
        raise NotImplementedError
