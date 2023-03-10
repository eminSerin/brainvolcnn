import torch.nn.functional as F
from torch import optim

try:
    from brainvolcnn.losses.loss_metric import corrcoef, r2_score
except ImportError:
    import os.path as op
    import sys

    path = op.abspath(op.join(op.dirname(__file__), op.join("..", "..")))
    if path not in sys.path:
        sys.path.append(path)
    del sys, path
    from losses.loss_metric import corrcoef, r2_score

from .utils import _activation_fn, _BaseLayer


class BaseModel(_BaseLayer):
    """Base class for CNN based semantic image segmentation architectures.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    max_level : int
        Maximum number of downsampling and upsampling levels except input, output and bottleneck layers.
    fdim : int, optional
        Number of features in the first convolutional layer,
        by default 64.
    n_conv : int, optional
        Number of convolutional layers in each stage, by default None.
    kernel_size : int, optional
        Convolutional kernel size, by default 3.
    padding : int, optional
        Convolutional padding, by default 1.
    stride : int, optional
        Convolutional stride, by default 1.
    activation : str, optional
        Activation function, by default "relu".
    up_mode : str, optional
        Upscaling method, by default "trilinear".
        It will be used if the input and target shapes are different.
    final_activation : str, optional
        Final activation function, by default None.
    loss_fn : function, optional
        Loss function, by default F.mse_loss.
    optimizer : function, optional
        Optimizer, by default optim.Adam.
    lr : float, optional
        Learning rate, by default 1e-3.
    """

    def __init__(
        self,
        in_chans,
        out_chans,
        max_level,
        fdim=64,
        n_conv=None,
        kernel_size=3,
        padding=1,
        stride=1,
        activation="relu_inplace",
        up_mode="trilinear",
        final_activation=None,
        loss_fn=F.mse_loss,
        optimizer=optim.Adam,
        lr=1e-3,
        add_loss={"corrcoef": corrcoef, "r2": r2_score},
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
        self.max_level = max_level
        self.fdim = fdim
        self.loss_fn = loss_fn
        self.add_loss = add_loss
        self.optimizer = optimizer
        self.lr = lr
        self._features = [fdim * 2**i for i in range(max_level + 1)]
        if final_activation is not None:
            self.final_activation = _activation_fn(final_activation, self.out_chans)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log("train_loss", loss)
        for name, fn in self.add_loss.items():
            self.log(f"train_{name}", fn(y_hat, y))
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        val_loss = self.loss_fn(y_hat, y)
        self.log("val_loss", val_loss)
        for name, fn in self.add_loss.items():
            self.log(f"val_{name}", fn(y_hat, y))
        return val_loss

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        return self(batch)

    # def test_step(self, batch, batch_idx):
    #     x, y = batch
    #     y_hat = self(x)
    #     test_loss = self.loss_fn(y_hat, y)
    #     self.log("test_loss", test_loss)
    #     self.log("test_r2", r2_score(y_hat, y))
    #     self.log("test_corr", corrcoef(y_hat, y))
    #     return test_loss

    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters(), lr=self.lr)
        # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": optim.lr_scheduler.ReduceLROnPlateau(optimizer),
                "monitor": "val_loss",
            },
        }
