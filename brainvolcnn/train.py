import os
import os.path as op
import sys

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

path = op.abspath(op.join(op.dirname(__file__), ".."))
if path not in sys.path:
    sys.path.append(path)
del sys, path
from brainvolcnn.callbacks.callbacks import (  # LogPredictionVariance,
    LogGradients,
    LogParameters,
    RCLossMarginTune,
    SaveLastModel,
)
from brainvolcnn.datasets.taskgen_dataset import TaskGenDataset
from brainvolcnn.losses.loss_metric import RCLossAnneal
from brainvolcnn.utils.parser import default_parser


def train(args):
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if op.exists(args.working_dir):
        raise FileExistsError(f"{args.working_dir} already exists!")
    else:
        os.makedirs(args.working_dir)

    """Load Data"""
    train_ids = np.genfromtxt(args.train_list, dtype=int, delimiter=",")
    if args.val_list is not None:
        val_ids = np.genfromtxt(args.val_list, dtype=int, delimiter=",")
    else:
        train_ids, val_ids = train_test_split(
            train_ids, test_size=args.val_percent, random_state=args.seed
        )

    unmask = False
    if args.mask is not None:
        unmask = True

    train_set = TaskGenDataset(
        train_ids,
        args.rest_dir,
        args.task_dir,
        num_samples=args.n_samples_per_subj,
        mask=args.mask,
        unmask=unmask,
    )
    val_set = TaskGenDataset(
        val_ids,
        args.rest_dir,
        args.task_dir,
        num_samples=args.n_samples_per_subj,
        mask=args.mask,
        unmask=unmask,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.n_workers,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.n_workers,
        drop_last=True,
    )

    """Init Model"""
    # Loads model from checkpoint if specified
    if args.checkpoint_file is not None:
        if not op.exists(args.checkpoint_file):
            raise FileNotFoundError(f"{args.checkpoint_file} does not exist!")
        model = args.architecture.load_from_checkpoint(
            args.checkpoint_file,
            in_chans=args.n_channels,
            out_chans=args.n_out_channels,
            fdim=args.fdim,
            activation=args.activation,
            final_activation=args.final_activation,
            optimizer=args.optimizer,
            up_mode=args.upsampling_mode,
            loss_fn=args.loss,
            add_loss=args.add_loss,
            max_level=args.max_depth,
            n_conv=args.n_conv_layers,
        )
    else:
        model = args.architecture(
            in_chans=args.n_channels,
            out_chans=args.n_out_channels,
            fdim=args.fdim,
            activation=args.activation,
            final_activation=args.final_activation,
            optimizer=args.optimizer,
            up_mode=args.upsampling_mode,
            loss_fn=args.loss,
            add_loss=args.add_loss,
            max_level=args.max_depth,
            n_conv=args.n_conv_layers,
        )

    """Checkpoint"""
    checkpoint_callback_loss = ModelCheckpoint(
        monitor="val/loss",
        dirpath=args.working_dir,
        filename="best_loss",
        save_top_k=1,
        mode="min",
        # save_last=True,
    )
    checkpoint_callback_r2 = ModelCheckpoint(
        monitor="val/r2",
        dirpath=args.working_dir,
        filename="best_r2",
        save_top_k=1,
        mode="max",
    )
    callbacks = [
        checkpoint_callback_loss,
        checkpoint_callback_r2,
        SaveLastModel(),
    ]

    # Logger
    if args.logger == "tensorboard":
        logger = TensorBoardLogger(
            args.working_dir, name="logs", version=args.ver, default_hp_metric=False
        )
        callbacks.extend([LogGradients(), LogParameters()])
    elif args.logger == "wandb":
        logger = WandbLogger(
            name=args.ver,
            project="brainvolcnn",
            config=args._hparams,
            save_dir=args.working_dir,
        )
        logger.watch(model, log="all", log_freq=50)
    logger.log_hyperparams(args._hparams)

    """Train Model"""
    ## TODO: Add early stopping!
    if isinstance(args.loss, RCLossAnneal):
        callbacks.append(RCLossMarginTune())

    trainer = pl.Trainer(
        max_epochs=args.n_epochs,
        accelerator=args.device,
        default_root_dir=args.working_dir,
        callbacks=callbacks,
        logger=logger,
        # limit_train_batches=1,
        # limit_val_batches=1,
    )
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)


if __name__ == "__main__":
    train(default_parser())
