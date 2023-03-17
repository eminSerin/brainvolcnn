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


def dice(input, target):
    raise NotImplementedError


def dice_auc(input, target):
    raise NotImplementedError


def _rc_loss_fn(input, target):
    """Computes the Reconstruction and Contrastive Losses.

    Reconstruction loss is the mean squared error between the input and target, aiming to minimize the difference between predicted and target values. Contrastive loss is the mean squared error between the input and the flipped target, aiming to maximize the difference between predicted values and target values of other subjects in a given batch.
    """
    if input.shape[0] < 2:
        raise ValueError(
            "Input and target must have at least 2 samples! Otherwise it cannot compute contrastive loss."
        )

    return FM.mean_squared_error(input, target), FM.mean_squared_error(
        input, torch.flip(target, dims=[0])
    )


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


def contrastive_loss(input, target, within_margin=1, between_margin=0.0, alpha=0.5):
    """Computes contrastive loss.

    It computes within and between contrastive loss and combines them using alpha.
    Within contrastive loss is computed as the mean of cosine similarity between input and target, while between contrastive loss is computed as the mean of cosine similarity between input and flipped target.

    Parameters
    ----------
    input : torch.Tensor
        Predicted values.
    target : torch.Tensor
        True values.
    within_margin : int, optional
        Ensure that within cosine similarity is below margin, by default 1
    between_margin : float, optional
        Ensure that between cosine similarity is above margin, by default 0.0
    alpha : float, optional
        Alpha to combine within and between loss. Specifically, alpha=1.0 is only within loss, while alpha=0 is between loss, by default 0.5

    Returns
    -------
    torch.float
        Contrastive loss.

    Raises
    ------
    ValueError
        Raise ValueError if input and target have different batch size.
    ValueError
        Raise ValueError if input and target have less than 2 samples.
    """
    # Compute contrastive loss using cosine similarity
    if input.shape[0] != target.shape[0]:
        raise ValueError("Input and target must have the same batch size")
    if input.shape[0] < 2:
        raise ValueError(
            "Input and target must have at least 2 samples! Otherwise it cannot compute contrastive loss."
        )
    within = F.cosine_similarity(input, target).mean()
    between = F.cosine_similarity(input, torch.flip(target, dims=[0])).mean()
    return (
        torch.clamp(within_margin - within, min=0) * alpha
        + torch.clamp(between - between_margin, min=0) * (1 - alpha)
    ) * 2


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
        self.mask = mask
        if mask is not None:
            self.mask = MaskTensor(mask)
        self.update_margins(epoch)

    # def update_margins(self, epoch):
    #     """Updates margins based on the current epoch."""
    #     self.within_margin = np.max(
    #         [
    #             self.init_within_margin * 0.5 ** (epoch // self.margin_anneal_step),
    #             self.min_within_margin,
    #         ]
    #     )
    #     self.between_margin = np.min(
    #         [
    #             self.init_between_margin * 2.0 ** (epoch // self.margin_anneal_step),
    #             self.max_between_margin,
    #         ],
    #     )

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

    def forward(self, input, target):
        if self.mask is not None:
            return rc_loss(
                self.mask.apply_mask(input),
                self.mask.apply_mask(target),
                within_margin=self.within_margin,
                between_margin=self.between_margin,
            )
        return rc_loss(
            input,
            target,
            within_margin=self.within_margin,
            between_margin=self.between_margin,
        )


class ContrastiveLoss(nn.Module):
    """Class for contrastive loss.

    Also see brainvolcnn.losses.loss_metric.contrastive_loss
    for more details.

    Arguments
    ----------
    mask : torch.Tensor, optional
        Mask tensor, by default None.
    alpha : float, optional
        Alpha value for combining within and between losses, by default 0.5
    within_margin : float, optional
        Within margin, by default 1.0
    between_margin : float, optional
        Between margin, by default 0.0
    """

    def __init__(self, mask=None, alpha=0.5, within_margin=1.0, between_margin=0.0):
        super().__init__()
        self.mask = mask
        if mask is not None:
            self.mask = MaskTensor(mask)
        self.alpha = alpha
        self.within_margin = within_margin
        self.between_margin = between_margin

    def forward(self, input, target):
        if self.mask is not None:
            return contrastive_loss(
                self.mask.apply_mask(input),
                self.mask.apply_mask(target),
                alpha=self.alpha,
                within_margin=self.within_margin,
                between_margin=self.between_margin,
            )
        return contrastive_loss(
            input,
            target,
            alpha=self.alpha,
            within_margin=self.within_margin,
            between_margin=self.between_margin,
        )


class ContrastiveLossAnneal(ContrastiveLoss):
    """Contrastive loss with annealing.

    Inherits from ContrastiveLoss. See ContrastiveLoss for more details.
    Despite ContrastiveLoss, this class has an additional parameter
    called anneal_step. This parameter controls the number of epochs
    required before the alpha value is annealed. During annealing the
    alpha value is decreased by a factor of anneal_percent. The minimum
    alpha value is defined by min_alpha.

    Methods:
    ----------
    update_alpha(epoch):
        Updates alpha value based on the current epoch.
    """

    def __init__(
        self, epoch=0, anneal_step=20, anneal_percent=0.1, min_alpha=0.0, **kwargs
    ):
        super().__init__(**kwargs)
        self.epoch = epoch
        self.anneal_step = anneal_step
        self.min_alpha = min_alpha
        self.anneal_percent = anneal_percent
        self._mod = None

    def update_alpha(self, epoch):
        _mod = epoch % self.anneal_step
        if _mod != 0 and _mod != self._mod:
            self._mod = _mod
            self.alpha = max(
                self.alpha - self.alpha * self.anneal_percent, self.min_alpha
            )
