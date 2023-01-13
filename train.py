import os.path as op

import numpy as np
import pytorch_lightning as pl
import torch
from callbacks.callbacks import RCLossMarginTune
from datasets.taskgen_dataset import TaskGenDataset
from losses.loss_metric import RCLossAnneal
from pytorch_lightning.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from utils.parser import default_parser


def train():
    args = default_parser()

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    ##TODO: Uncomment this
    # if op.exists(args.working_dir):
    #     raise FileExistsError(f"{args.working_dir} already exists!")
    # else:
    #     op.makedirs(args.working_dir)

    """Load Data"""
    subj_ids = np.genfromtxt(args.subj_list, dtype=int, delimiter=",")
    train_ids, val_ids = train_test_split(
        subj_ids, test_size=args.val_percent, random_state=args.seed
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
        device=args.device,
    )
    val_set = TaskGenDataset(
        val_ids,
        args.rest_dir,
        args.task_dir,
        num_samples=args.n_samples_per_subj,
        mask=args.mask,
        unmask=unmask,
        device=args.device,
    )

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)

    """Init Model"""
    model = args.architecture(
        in_chans=args.n_channels,
        out_chans=args.n_out_channels,
        fdim=args.fdim,
        activation=args.activation,
        final_activation=args.final_activation,
        optimizer=args.optimizer,
        up_mode=args.upsampling_mode,
        loss_fn=args.loss,
    )

    """Checkpoint"""
    checkpoint_callback_loss = ModelCheckpoint(
        monitor="val_loss",
        dirpath=args.working_dir,
        filename="best_loss",
        save_top_k=2,
        mode="min",
    )
    checkpoint_callback_r2 = ModelCheckpoint(
        monitor="val_r2",
        dirpath=args.working_dir,
        filename="best_r2",
        save_top_k=2,
        mode="max",
    )
    callbacks = [checkpoint_callback_loss, checkpoint_callback_r2]

    """Train Model"""
    ## TODO: Add early stopping!
    if isinstance(args.loss, RCLossAnneal):
        callbacks.append(RCLossMarginTune())

    trainer = pl.Trainer(
        max_epochs=args.n_epochs,
        accelerator=args.device,
        default_root_dir=args.working_dir,
        callbacks=callbacks,
    )
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)


if __name__ == "__main__":
    train()