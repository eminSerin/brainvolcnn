import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets.utils import MaskTensor
from torchmetrics import functional as FM

##TODO: Add docstrings!

"""Functional"""


def r2_score(input, target):
    """Wrapper for torchmetrics.functional.r2_score"""
    return FM.r2_score(input.flatten(), target.flatten())


def r2_loss(input, target):
    """Loss function for R2 score."""
    return 1 - r2_score(input, target)


def corrcoef(input, target):
    """Wrapper for torchmetrics.functional.pearson_corrcoef"""
    return FM.pearson_corrcoef(input.flatten(), target.flatten())


def corrcoef_loss(input, target):
    """Loss function for Pearson correlation coefficient."""
    return 1 - corrcoef(input, target)


def between_mse(input, target):
    """Computes the mean squared error between the input and the flipped target, aiming to maximize the difference between predicted values and target values of other subjects in a given batch"""
    return FM.mean_squared_error(input, torch.flip(target, dims=[0]))


def dice(input, target):
    raise NotImplementedError


def dice_auc(input, target):
    raise NotImplementedError


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
    if input.shape[0] < 2:
        raise ValueError(
            "Input and target must have at least 2 samples! Otherwise it cannot compute contrastive loss."
        )
    recon_loss = FM.mean_squared_error(input, target)
    contrast_loss = FM.mean_squared_error(input, torch.flip(target, dims=[0]))
    return torch.clamp(recon_loss - within_margin, min=0.0) + torch.clamp(
        recon_loss - contrast_loss + between_margin, min=0.0
    )


"""OOP"""


class BaseLoss(nn.Module):
    def __init__(self, mask=None, loss_fn=None):
        super().__init__()
        self.mask = mask
        if mask is not None:
            self.mask = MaskTensor(mask)
        self.loss_fn = loss_fn

    def forward(self, input, target):
        if self.mask is not None:
            return self.loss_fn(
                self.mask.apply_mask(input), self.mask.apply_mask(target)
            )
        return self.loss_fn(input, target)


class MSELoss(BaseLoss):
    def __init__(self, mask=None):
        super().__init__(mask=mask, loss_fn=nn.functional.mse_loss)


class BetweenMSELoss(BaseLoss):
    def __init__(self, mask=None):
        super().__init__(mask=mask, loss_fn=between_mse)


class MAELoss(BaseLoss):
    def __init__(self, mask=None):
        super().__init__(mask=mask, loss_fn=nn.functional.l1_loss)


class HuberLoss(BaseLoss):
    def __init__(self, mask=None):
        super().__init__(mask=mask, loss_fn=nn.functional.huber_loss)


class MSLELoss(BaseLoss):
    def __init__(self, mask=None):
        super().__init__(mask=mask, loss_fn=FM.mean_squared_log_error)


class PearsonCorr(nn.Module):
    def __init__(self, loss=False, mask=None):
        super().__init__()
        self.loss = loss
        self.mask = mask
        if mask is not None:
            self.mask = MaskTensor(mask)

    def forward(self, input, target):
        if self.mask is not None:
            input = self.mask.apply_mask(input)
            target = self.mask.apply_mask(target)
        if self.loss:
            return corrcoef_loss(input, target)
        return corrcoef(input, target)


class R2(nn.Module):
    def __init__(self, loss=False, mask=None):
        super().__init__()
        self.loss = loss
        self.mask = mask
        if mask is not None:
            self.mask = MaskTensor(mask)

    def forward(self, input, target):
        if self.mask is not None:
            input = self.mask.apply_mask(input)
            target = self.mask.apply_mask(target)
        if self.loss:
            return r2_loss(input, target)
        return r2_score(input, target)


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
    alpha : float, optional
        The weight of the reconstructive loss, by default 0.5
        1.0: only reconstructive loss
    mask : torch.Tensor, optional
        Mask tensor, by default None.

    Returns
    ----------

    torch.float:
        RC Loss between target and input.
    """

    def __init__(
        self,
        epoch=0,
        init_within_margin=4.0,
        init_between_margin=5,
        min_within_margin=1.0,
        max_between_margin=10,
        margin_anneal_step=10,
        alpha=0.5,
        mask=None,
    ):
        super().__init__()
        self.init_within_margin = init_within_margin
        self.init_between_margin = init_between_margin
        self.min_within_margin = min_within_margin
        self.max_between_margin = max_between_margin
        self.margin_anneal_step = margin_anneal_step
        self.within_margin = init_within_margin
        self.between_margin = init_between_margin
        self.alpha = alpha
        self.mask = mask
        if mask is not None:
            self.mask = MaskTensor(mask)
        self.update_margins(epoch)

    def update_margins(self, epoch):
        if (epoch % self.margin_anneal_step == 0) & (epoch > 0):
            self.within_margin = np.max(
                [
                    self.within_margin - self.within_margin * 0.1,
                    self.min_within_margin,
                ]
            )
            self.between_margin = np.min(
                [
                    self.between_margin + self.between_margin * 0.1,
                    self.max_between_margin,
                ],
            )

    def _rc_loss(self, input, target, within_margin, between_margin):
        """Reconstruction and Contrastive Loss.

        See brainvolcnn.loss.rc_loss for more details.
        """
        self.recon_loss = FM.mean_squared_error(input, target)
        self.contrast_loss = FM.mean_squared_error(input, torch.flip(target, dims=[0]))
        return (
            torch.clamp(self.recon_loss - within_margin, min=0.0) * self.alpha
            + torch.clamp(
                self.recon_loss - self.contrast_loss + between_margin, min=0.0
            )
            * (1 - self.alpha)
            * 2
        )

    def forward(self, input, target):
        if self.mask is not None:
            return self._rc_loss(
                self.mask.apply_mask(input),
                self.mask.apply_mask(target),
                within_margin=self.within_margin,
                between_margin=self.between_margin,
            )
        return self._rc_loss(
            input,
            target,
            within_margin=self.within_margin,
            between_margin=self.between_margin,
        )
