import sys
import warnings

import numpy as np
import torch
import torch.nn as nn

try:
    from brainvolcnn.losses.loss_metric import corrcoef, r2_score, vae_loss
except ImportError:
    import os.path as op
    import sys

    path = op.abspath(op.join(op.dirname(__file__), op.join("..", "..")))
    if path not in sys.path:
        sys.path.append(path)
    del sys, path
    from losses.loss_metric import corrcoef, r2_score, vae_loss

from .base_model import BaseModel
from .utils import _BaseLayer, _interpolate, _skip_add, call_layer


class EncodeLayer(_BaseLayer):
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
        residual=True,
        batch_norm=True,
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
        self.residual = residual
        self.batch_norm = batch_norm
        if n_conv is None:
            self.n_conv = 2
        layers = nn.ModuleList()
        in_ch = self.in_chans
        for _ in range(self.n_conv):
            layers.append(
                call_layer("Conv", dims)(
                    in_ch,
                    self.out_chans,
                    kernel_size=self.kernel_size,
                    padding=self.padding,
                    stride=self.stride,
                )
            )
            in_ch = self.out_chans
            if self.batch_norm:
                layers.append(call_layer("BatchNorm", dims)(in_ch))
            layers.append(self._activation_fn)

        self.conv = nn.Sequential(*layers)

        # Skip connection
        if self.residual:
            self.skip = nn.Sequential()
            if self.in_chans != self.out_chans:
                self.skip = nn.Sequential(
                    (
                        call_layer("Conv", dims)(
                            self.in_chans,
                            self.out_chans,
                            kernel_size=1,
                            padding=0,
                            stride=1,
                        )
                    )
                )
        self.pool = call_layer("AvgPool", dims)(kernel_size=2, stride=2)

    def forward(self, x):
        if self.residual:
            return self.pool(_skip_add(self.conv(x), self.skip(x), mode=self.up_mode))
        return self.pool(self.conv(x))


class DecodeLayer(_BaseLayer):
    """Transposed convolutional layer.

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
        residual=True,
        batch_norm=True,
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
        self.residual = residual
        self.batch_norm = batch_norm
        if n_conv is None:
            self.n_conv = 2
        layers = nn.ModuleList()
        in_ch = self.in_chans
        for i in range(self.n_conv):
            if i == 0:  # first layer
                conv = call_layer("ConvTranspose", dims)(
                    in_ch,
                    self.out_chans,
                    kernel_size=self.kernel_size,
                    padding=self.padding,
                    stride=2,
                )
            else:
                conv = call_layer("Conv", dims)(
                    in_ch,
                    self.out_chans,
                    kernel_size=self.kernel_size,
                    padding=self.padding,
                    stride=self.stride,
                )
            layers.append(conv)
            in_ch = self.out_chans
            if self.batch_norm:
                layers.append(call_layer("BatchNorm", dims)(in_ch))
            layers.append(self._activation_fn)

        self.conv_transpose = nn.Sequential(*layers)

        # Skip connection
        if self.residual:
            self.skip = nn.Sequential()
            if self.in_chans != self.out_chans:
                self.skip = nn.Sequential(
                    (
                        call_layer("ConvTranspose", dims)(
                            self.in_chans,
                            self.out_chans,
                            kernel_size=1,
                            padding=0,
                            stride=2,
                        )
                    )
                )

    def forward(self, x):
        if self.residual:
            return _skip_add(self.conv_transpose(x), self.skip(x), mode=self.up_mode)
        return self.conv_transpose(x)


class _BaseVAE(BaseModel):
    def __init__(
        self,
        in_chans,
        out_chans,
        input_shape=None,
        max_level=2,
        dims=3,
        fdim=32,
        latent_dim=128,
        n_conv=None,
        kernel_size=3,
        padding=1,
        stride=1,
        residual=False,
        activation="relu_inplace",
        final_activation=False,
        up_mode="trilinear",
        loss_fn=vae_loss,
        optimizer=torch.optim.Adam,
        lr=1e-3,
        add_loss={"corrcoef": corrcoef, "r2": r2_score},
        batch_norm=True,
        lr_scheduler=True,
        **kwargs,
    ):
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
            None,
            vae_loss,
            optimizer,
            lr,
            add_loss,
            batch_norm,
            lr_scheduler,
            **kwargs,
        )
        warnings.warn("This model is not fully tested yet.")
        warnings.warn("VAE will only be trained using vae_loss function.")
        if input_shape is None:
            raise ValueError("input_shape must be specified.")
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.residual = residual
        self.dims = dims
        self.encoder_layers = nn.ModuleList()
        self.decoder_layers = nn.ModuleList()

        # Encoder layers
        in_dim = in_chans
        for feat in self._features:
            self.encoder_layers.append(
                EncodeLayer(
                    in_dim,
                    feat,
                    n_conv=n_conv,
                    dims=self.dims,
                    kernel_size=kernel_size,
                    padding=padding,
                    stride=stride,
                    activation=activation,
                    up_mode=up_mode,
                    residual=residual,
                    batch_norm=batch_norm,
                )
            )
            in_dim = feat
        self.encoder_layers = nn.Sequential(*self.encoder_layers)
        self._encoder_shapes = self._infer_input_shape
        self._last_conv_shape = self._encoder_shapes[-1]
        self._last_conv_flat_shape = np.prod(self._last_conv_shape)

        # Latent space
        self.fc_mu = nn.Linear(self._last_conv_flat_shape, latent_dim)
        self.fc_var = nn.Linear(self._last_conv_flat_shape, latent_dim)

        # Decoding Layers
        in_dim = feat
        self.decoder_input_layer = nn.Linear(latent_dim, self._last_conv_flat_shape)
        for feat in reversed(self._features[:-1]):
            self.decoder_layers.append(
                DecodeLayer(
                    in_dim,
                    feat,
                    n_conv=n_conv,
                    dims=self.dims,
                    kernel_size=kernel_size,
                    padding=padding,
                    stride=stride,
                    activation=activation,
                    up_mode=up_mode,
                    residual=False,
                    batch_norm=batch_norm,
                )
            )
            in_dim = feat
        self.decoder_layers.append(
            DecodeLayer(
                feat,
                feat,
                n_conv=n_conv,
                dims=dims,
                kernel_size=kernel_size,
                padding=padding,
                stride=stride,
                activation=activation,
                up_mode=up_mode,
                residual=False,
                batch_norm=batch_norm,
            )
        )
        # Output Layer
        self.final_layer = call_layer("Conv", dims)(
            feat, out_chans, kernel_size=1, padding=0, stride=1
        )
        self.decoder_layers = nn.Sequential(*self.decoder_layers)

    @property
    def _infer_input_shape(self):
        # Generate a random tensor
        with torch.no_grad():
            x = torch.randn(1, self.in_chans, *self.input_shape)
            x_shapes = []
            # Pass it through the encoder
            for layer in self.encoder_layers:
                x = layer(x)
                x_shapes.append(x.shape[1:])
        return x_shapes

    def encode(self, x):
        x = torch.flatten(self.encoder_layers(x), start_dim=1)
        return self.fc_mu(x), self.fc_var(x)

    def decode(self, z):
        z = self.decoder_input_layer(z).view(-1, *self._last_conv_shape)
        reversed_shapes = self._encoder_shapes[::-1][1:] + [(1, *self.input_shape)]
        for shape, layer in zip(reversed_shapes, self.decoder_layers):
            z = _interpolate(layer(z), shape[1:], mode=self.up_mode)
        return self.final_layer(z)
        # return self.final_layer(
        #     _interpolate(
        #         self.decoder_layers(
        #             self.decoder_input_layer(z).view(-1, *self._last_conv_shape)
        #         ),
        #         self.input_shape,
        #         mode=self.up_mode,
        #     )
        # )

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, log_var = self.encode(x)
        return self.decode(self.reparameterize(mu, log_var)), mu, log_var

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat, mu, log_var = self(x)
        loss = self.loss_fn(y_hat, y, mu, log_var)
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=False)
        for name, fn in self.add_loss.items():
            self.log(
                f"train/{name}",
                fn(y_hat, y),
                prog_bar=True,
                on_step=True,
                on_epoch=False,
            )
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat, mu, log_var = self(x)
        val_loss = self.loss_fn(y_hat, y, mu, log_var)
        self.log("val/loss", val_loss, on_step=False, on_epoch=True)
        for name, fn in self.add_loss.items():
            self.log(f"val/{name}", fn(y_hat, y), on_step=False, on_epoch=True)
        return val_loss


class VAE3D(_BaseVAE):
    def __init__(
        self,
        *args,
        input_shape=(76, 93, 78),
        dims=3,
        **kwargs,
    ):
        super().__init__(
            input_shape=input_shape,
            *args,
            dims=dims,
            **kwargs,
        )
