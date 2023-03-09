import os
import os.path as op

import numpy as np
import torch
from tqdm import tqdm

try:
    from brainvolcnn.datasets.taskgen_dataset import load_timeseries
    from brainvolcnn.utils.parser import default_parser
except ImportError:
    import sys

    path = op.abspath(op.join(op.dirname(__file__), ".."))
    if path not in sys.path:
        sys.path.append(path)
    del sys, path
    from brainvolcnn.datasets.taskgen_dataset import load_timeseries
    from brainvolcnn.utils.parser import default_parser


def predict(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if op.exists(args.working_dir):
        raise FileExistsError(f"{args.working_dir} already exists!")
    else:
        os.makedirs(args.working_dir)

    """Load Datalist"""
    subj_ids = np.genfromtxt(args.test_list, dtype=int, delimiter=",")

    """Init Model"""
    if args.checkpoint_file is None:
        raise ValueError("Must provide a checkpoint to load!")
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
    ).to(args.device)

    """Predict"""
    print("Predicting...")
    model.eval()
    with torch.no_grad():
        with tqdm(total=len(subj_ids)) as pbar:
            for id in subj_ids:
                pbar.set_description(f"Predicting {id}...")
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
                            device=args.device,
                        )
                        pred_list.append(
                            model(img.unsqueeze(0)).cpu().detach().numpy().squeeze(0)
                        )
                    np.save(pred_file, np.array(pred_list))
                else:
                    print(f"Skipping {id} because prediction already exists!")

    print("Finished predicting!")


if __name__ == "__main__":
    predict(default_parser())
