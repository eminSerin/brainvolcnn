"""Main input parser for BrainVolCNN"""

from argparse import ArgumentParser

import torch
from losses.loss_metric import RCLossAnneal


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
        "--subj_list",
        type=str,
        help="File containing the subject ID list, one subject ID on each line",
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
        choices=["resunet", "unet", "vnet"],
        default="resunet",
        help="Model architecture",
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
        choices=["rc", "mse"],
        default="mse",
        help="Loss function, default=mse",
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
        choices=["trilinear", "nearest"],
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
        help="Percent of training data for validation",
    )

    parser.add_argument(
        "--init_within_subj_margin",
        type=float,
        default=4.0,
        help="Initial within-subject margin, which should be computed on the training set",
    )

    parser.add_argument(
        "--min_within_subj_margin",
        type=float,
        default=1.0,
        help="Minimum within-subject margin that was aimed for",
    )

    parser.add_argument(
        "--init_across_subj_margin",
        type=float,
        default=5.0,
        help="Initial across-subject margin, which should be computed on the training set",
    )

    parser.add_argument(
        "--max_across_subj_margin",
        type=float,
        default=10.0,
        help="Maximum across-subject margin that was aimed for",
    )

    parser.add_argument(
        "--margin_anneal_step",
        type=int,
        default=10,
        help="Step for annealing the margins",
    )

    parser.add_argument(
        "--mask",
        type=str,
        help="Mask to use reconstruct fMRI images, if extracted timeseries were provided.",
    )

    parser.add_argument("--checkpoint_file", type=str, help="Path to checkpoint file")

    args = parser.parse_args()

    if args.mask is not None:
        args.unmask = True
    else:
        args.unmask = False

    # Loss
    if args.loss == "mse":
        args.loss = torch.nn.MSELoss()
    elif args.loss == "rc":
        args.loss = RCLossAnneal(
            init_within_margin=args.init_within_subj_margin,
            init_between_margin=args.init_across_subj_margin,
            min_within_margin=args.min_within_subj_margin,
            max_between_margin=args.max_across_subj_margin,
            margin_anneal_step=args.margin_anneal_step,
        )
    else:
        raise ValueError(f"Loss {args.loss} not supported")

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
        from models.res_unet import ResUNet

        args.architecture = ResUNet
    elif args.architecture == "unet":
        from models.unet import UNet

        args.architecture = UNet
    elif args.architecture == "vnet":
        from models.vnet import VNet

        args.architecture = VNet
    return args
