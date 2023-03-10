"""Main input parser for BrainVolCNN"""

from argparse import ArgumentParser

import torch

from brainvolcnn import models
from brainvolcnn.losses.loss_metric import (
    R2,
    ContrastiveLoss,
    ContrastiveLossAnneal,
    HuberLoss,
    MAELoss,
    MSELoss,
    MSLELoss,
    PearsonCorr,
    RCLossAnneal,
)


def default_parser():
    parser = ArgumentParser("BrainVolCNN")

    parser.add_argument("--rest_dir", type=str, help="Path to resting-state data")

    parser.add_argument("--task_dir", type=str, help="Path to task data")

    parser.add_argument(
        "--working_dir", type=str, help="Path to working directory (for saving models)"
    )

    parser.add_argument(
        "--ver", type=str, help="Additional string for the name of the file"
    )

    parser.add_argument(
        "--train_list",
        type=str,
        help="File containing the train subject ID list, one subject ID on each line",
    )

    parser.add_argument(
        "--val_list",
        type=str,
        help="File containing the validation subject ID list, one subject ID on each line",
    )

    parser.add_argument(
        "--test_list",
        type=str,
        help="File containing the test subject ID list, one subject ID on each line",
    )

    parser.add_argument(
        "--n_samples_per_subj",
        type=int,
        default=8,
        help="Number of rsfc samples per subject, default=8",
    )

    parser.add_argument(
        "--architecture",
        type=str,
        choices=["resunet", "unet"],
        default="resunet",
        help="Model architecture",
    )

    parser.add_argument(
        "--conv_dim",
        type=int,
        choices=[1, 3],
        default=3,
        help="Dimension of convolutional kernels, default=3",
    )

    parser.add_argument(
        "--n_conv_layers",
        type=int,
        default=3,
        help="Number of convolutional layers in each block",
    )

    parser.add_argument("--batch_size", type=int, default=6, help="Batch size")

    parser.add_argument(
        "--n_channels",
        type=int,
        default=50,
        help="Number of input channels, default=50",
    )

    parser.add_argument(
        "--n_out_channels",
        type=int,
        default=47,
        help="Number of output channels, default=47",
    )

    parser.add_argument(
        "--max_depth",
        type=int,
        default=5,
        help="Maximum depth of encoder and decoder networks",
    )

    parser.add_argument(
        "--fdim",
        type=int,
        default=64,
        help="Number of feature dimensions (i.e., features in the first convolutional layer), default=64",
    )

    parser.add_argument("--n_epochs", type=int, default=100, help="Number of epochs")

    parser.add_argument(
        "--activation",
        type=str,
        choices=[
            "relu",
            "relu_inplace",
            "leakyrelu",
            "leakyrelu_inplace",
            "elu",
            "elu_inplace",
            "tanh",
            "prelu",
        ],
        default="relu_inplace",
        help="Activation function, default=relu_inplace",
    )

    parser.add_argument(
        "--final_activation",
        type=str,
        choices=["relu", "relu_inplace", "prelu", "softmax", "sigmoid"],
        default="relu_inplace",
        help="Final activation function, default=relu_inplace",
    )

    parser.add_argument(
        "--loss",
        type=str,
        choices=[
            "rc",
            "mse",
            "contrastive",
            "contrastive_anneal",
            "mae",
            "msle",
            "huber",
        ],
        default="mse",
        help="Loss function, default=mse",
    )

    parser.add_argument(
        "--loss_mask",
        type=str,
        default=None,
        help="Mask to use for loss computation, default=None",
    )

    parser.add_argument(
        "--optimizer",
        type=str,
        choices=["adam", "lbfgs", "sgd"],
        default="adam",
        help="Optimizer, default=adam",
    )

    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")

    parser.add_argument(
        "--upsampling_mode",
        type=str,
        choices=["trilinear", "nearest", "linear"],
        default="trilinear",
        help="Upsampling mode, default=trilinear",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cuda", "cpu", "tpu", "mps"],
        help="Device to run analyses on, default=cuda",
    )

    parser.add_argument(
        "--n_workers", type=int, default=4, help="Number of workers for dataloader"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for numpy to create train/val split",
    )

    parser.add_argument(
        "--val_percent",
        type=float,
        default=0.1,
        help="Percent of training data for validation. This is only used if val_list is not provided",
    )

    parser.add_argument(
        "--init_within_subj_margin",
        type=float,
        default=0.5,
        help="Initial within-subject margin, which should be computed on the training set",
    )

    parser.add_argument(
        "--min_within_subj_margin",
        type=float,
        default=0.01,
        help="Minimum within-subject margin that was aimed for",
    )

    parser.add_argument(
        "--init_across_subj_margin",
        type=float,
        default=0.6,
        help="Initial across-subject margin, which should be computed on the training set",
    )

    parser.add_argument(
        "--max_across_subj_margin",
        type=float,
        default=1,
        help="Maximum across-subject margin that was aimed for",
    )

    parser.add_argument(
        "--anneal_step",
        type=int,
        default=10,
        help="Step for annealing loss function parameters",
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Alpha weight for contrastive loss, default=0.5",
    )

    parser.add_argument(
        "--min_alpha",
        type=float,
        default=0.0,
        help="Minimum alpha weight for contrastive loss, default=0.0",
    )

    parser.add_argument(
        "--mask",
        type=str,
        help="Mask to use reconstruct fMRI images, if extracted timeseries were provided.",
    )

    parser.add_argument("--checkpoint_file", type=str, help="Path to checkpoint file")

    parser.add_argument(
        "--logger",
        type=str,
        default="tensorboard",
        help="Logger to use, default=tensorboard",
    )

    args = parser.parse_args()

    if args.mask is not None:
        args.unmask = True
    else:
        args.unmask = False

    args._hparams = {
        "conv_dim": args.conv_dim,
        "max_depth": args.max_depth,
        "loss": args.loss,
        "optimizer": args.optimizer,
        "lr": args.lr,
        "architecture": args.architecture,
        "activation": args.activation,
        "n_epochs": args.n_epochs,
        "upsampling_mode": args.upsampling_mode,
        "n_conv_layers": args.n_conv_layers,
    }

    # Loss
    if args.loss == "mse":
        args.loss = MSELoss(mask=args.loss_mask)
    elif args.loss == "msle":
        args.loss = MSLELoss(mask=args.loss_mask)
    elif args.loss == "mae":
        args.loss = MAELoss(mask=args.loss_mask)
    elif args.loss == "huber":
        args.loss = HuberLoss(mask=args.loss_mask)
    elif args.loss == "rc":
        args.loss = RCLossAnneal(
            init_within_margin=args.init_within_subj_margin,
            init_between_margin=args.init_across_subj_margin,
            min_within_margin=args.min_within_subj_margin,
            max_between_margin=args.max_across_subj_margin,
            margin_anneal_step=args.anneal_step,
            mask=args.loss_mask,
        )
    elif args.loss == "contrastive":
        args.loss = ContrastiveLoss(
            alpha=args.alpha,
            mask=args.loss_mask,
        )
    elif args.loss == "contrastive_anneal":
        ## TODO: Add anneal percent for annealing. Not urgent.
        args.loss = ContrastiveLossAnneal(
            alpha=args.alpha,
            min_alpha=args.min_alpha,
            within_margin=args.init_within_subj_margin,
            between_margin=args.init_across_subj_margin,
            margin_anneal_step=args.anneal_step,
            mask=args.loss_mask,
        )
    else:
        raise ValueError(f"Loss {args.loss} not supported")

    args.add_loss = {
        "corr": PearsonCorr(mask=args.loss_mask),
        "r2": R2(mask=args.loss_mask),
    }

    # Version
    if args.ver is None:
        args.ver = f"{args.architecture}_{args.n_conv_layers}_{args.lr}_{args.loss}_{args.optimizer}_{args.activation}"

    # Optimizer
    if args.optimizer == "adam":
        args.optimizer = torch.optim.Adam
    elif args.optimizer == "lbfgs":
        args.optimizer = torch.optim.LBFGS
    elif args.optimizer == "sgd":
        args.optimizer = torch.optim.SGD
    else:
        raise ValueError(f"Optimizer {args.optimizer} not supported")

    # Architecture
    if args.architecture == "resunet":
        args.architecture = getattr(models.resunet, f"ResUNet{args.conv_dim}D")
    elif args.architecture == "unet":
        args.architecture = getattr(models.unet, f"UNet{args.conv_dim}D")
    elif args.architecture == "vnet":
        args.architecture = getattr(models.vnet, f"VNet{args.conv_dim}D")
    return args
