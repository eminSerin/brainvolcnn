import numpy as np
import torch
import torch.nn as nn
from torchmetrics import functional as FM

##TODO: Add docstrings!


def _batch_mse(input, target):
    """Computes the mean squared error between two tensors.

    Parameters
    ----------
    input : _type_
        _description_
    target : _type_
        _description_

    Returns
    -------
    _type_
        _description_

    Raises
    ------
    ValueError
        _description_
    """
    if input.shape != target.shape:
        raise ValueError("Input and target shapes must match!")
    return torch.mean((input.flatten(1) - target.flatten(1)) ** 2, dim=-1)


def _rc_loss_fn(input, target):
    """Computes the Reconstruction and Contrastive Losses.

    Reconstruction loss is the mean squared error between the input and target, aiming to minimize the difference between predicted and target values. Contrastive loss is the mean squared error between the input and the flipped target, aiming to maximize the difference between predicted values and target values of other subjects in a given batch.
    """
    if input.shape[0] < 2:
        raise ValueError(
            "Input and target must have at least 2 samples! Otherwise it cannot compute contrastive loss."
        )

    # Reconstruction loss (i.e., MSE)
    recon_loss = _batch_mse(input, target)
    # Contrastive loss
    contrast_loss = _batch_mse(input, torch.flip(target, dims=[0]))
    return torch.mean(recon_loss), torch.mean(contrast_loss)


def rc_loss(input, target, within_margin=0, between_margin=0):
    """Construction Reconstruction Loss (RC Loss) as described in [1].

    Parameters
    ----------
    input : torch.Tensor
        Predicted values.
    target : torch.Tensor
        Target values.
    within_margin : int, optional
        Same subject (reconstructive) margin, by default 0.
    between_margin : int, optional
        Between subject (contrastive) margin, by default 0.

    Returns
    -------
    torch.float
        RC Loss.

    References:
    -----------
    [1] Ngo, Gia H., et al. "Predicting individual task contrasts from resting‐state functional connectivity using a surface‐based convolutional network." NeuroImage 248 (2022): 118849.
    """
    recon_loss, contrast_loss = _rc_loss_fn(input, target)
    return torch.clamp(recon_loss - within_margin, min=0.0) + torch.clamp(
        recon_loss - contrast_loss + between_margin, min=0.0
    )


class RCLossAnneal(nn.Module):
    """Reconstruction and Contrastive Loss with Annealing.

    Parameters
    ----------
    epoch : int, optional
        The current epoch, by default 0
    init_within_margin : int, optional
        Initial same subject (reconstructive) margin, by default 4
    init_between_margin : int, optional
        Initial between subject (contrastive) margin, by default 5
    min_within_margin : int, optional
        Minimum same subject (reconstructive) margin, by default 1
    max_between_margin : int, optional
        Maximum between subject (contrastive) margin, by default 10
    margin_anneal_step : int, optional
        The number of epochs should be done before margin annealing happens, by default 10

    Returns
    ----------
    torch.float:
        RC Loss between target and input.
    """

    def __init__(
        self,
        epoch=0,
        init_within_margin=4,
        init_between_margin=5,
        min_within_margin=1,
        max_between_margin=10,
        margin_anneal_step=10,
    ):
        super().__init__()
        self.init_within_margin = init_within_margin
        self.init_between_margin = init_between_margin
        self.min_within_margin = min_within_margin
        self.max_between_margin = max_between_margin
        self.margin_anneal_step = margin_anneal_step
        self.update_margins(epoch)

    def update_margins(self, epoch):
        """Updates margins based on the current epoch."""
        self.within_margin = np.max(
            [
                self.init_within_margin * 0.5 ** (epoch // self.margin_anneal_step),
                self.min_within_margin,
            ]
        )
        self.between_margin = np.min(
            [
                self.init_between_margin * 2.0 ** (epoch // self.margin_anneal_step),
                self.max_between_margin,
            ],
        )

    def forward(self, input, target):
        return rc_loss(input, target, self.within_margin, self.between_margin)


def r2_score(input, target):
    """Wrapper for torchmetrics.functional.r2_score"""
    return FM.r2_score(input.flatten(), target.flatten())


def corrcoef(input, target):
    """Wrapper for torchmetrics.functional.pearson_corrcoef"""
    return FM.pearson_corrcoef(input.flatten(), target.flatten())


def dice(input, target):
    raise NotImplementedError


def dice_auc(input, target):
    raise NotImplementedError
