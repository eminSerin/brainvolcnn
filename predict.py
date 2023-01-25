import os.path as op

import numpy as np
import pytorch_lightning as pl
import torch
from datasets.taskgen_dataset import load_timeseries
from tqdm import tqdm
from utils.parser import default_parser


def predict():
    args = default_parser()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if op.exists(args.working_dir):
        raise FileExistsError(f"{args.working_dir} already exists!")
    else:
        op.makedirs(args.working_dir)

    """Load Datalist"""
    subj_ids = np.genfromtxt(args.subj_list, dtype=int, delimiter=",")

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
    if args.checkpoint is None:
        raise ValueError("Must provide a checkpoint to load!")
    model.load_from_checkpoint(args.checkpoint_file)

    """Predict"""
    print("Predicting...")
    model.eval()
    with torch.no_grad():
        with tqdm(total=len(subj_ids)) as pbar:
            for i, id in enumerate(subj_ids):
                pbar.set_description(f"Predicting {subj_ids[i]}...")
                pbar.update(1)
                pred_file = op.join(args.working_dir, f"{id}_pred.npy")
                if not op.exists(pred_file):
                    pred_list = []
                    for sample_id in range(args.n_samples_per_subj):
                        rest_file = op.join(
                            args.rest_dir, f"{id}_sample{sample_id}_rsfc.npy"
                        )
                        img = load_timeseries(
                            rest_file,
                            mask=args.mask,
                            unmask=args.unmask,
                            crop=args.crop,
                            device=args.device,
                        )
                        pred_list.append(model(img).cpu().detach().numpy().squeeze(0))
                    np.save(pred_file, np.array(pred_list))
                else:
                    print(f"Skipping {id} because prediction already exists!")

    print("Finished predicting!")
