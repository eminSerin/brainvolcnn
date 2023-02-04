import os
from argparse import ArgumentParser

import numpy as np
from joblib import Parallel, delayed
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

parser = ArgumentParser()

parser.add_argument(
    "--subj_list",
    type=str,
    help="File containing the subject ID list, one subject ID on each line",
)

parser.add_argument("--pred_dir", type=str, help="Path to the output directory")

parser.add_argument(
    "--target_dir",
    type=str,
    help="Directory containing the subject target task contrast files for training and validation",
)

parser.add_argument(
    "--batch_size",
    type=int,
    default=10,
    help="Number of subjects used to compute the within/across subject loss",
)

parser.add_argument("--n_jobs", type=int, default=1, help="Number of jobs")

parser.add_argument("--seed", type=int, default=42, help="Random seed")


def compute_loss(pred, target):
    """Compute the mean squared error between the predicted and target volumes."""
    return mean_squared_error(target, pred)


def compute_losses_across_subj(subj, subj_ids, pred_dir, true_dir, batch_size):
    batch_ids = np.random.choice(subj_ids, batch_size + 1, replace=False)
    other_ids = np.setdiff1d(batch_ids, np.asarray([subj]))[:batch_size]

    pred = np.load(os.path.join(pred_dir, f"{subj}_pred.npy")).mean(axis=0).flatten()
    target = np.load(
        os.path.join(true_dir, f"{subj}_joint_MNI_task_contrasts.npy")
    ).flatten()
    within_mse = compute_loss(pred, target)

    across_mse = 0
    for other in other_ids:
        target = np.load(
            os.path.join(true_dir, f"{other}_joint_MNI_task_contrasts.npy")
        ).flatten()
        across_mse += compute_loss(pred, target) / (len(other_ids))
    return within_mse, across_mse


def main(args):
    subj_ids = np.genfromtxt(args.subj_list, dtype=int, delimiter=",")
    within_mse = 0
    across_mse = 0
    within_mse, across_mse = Parallel(n_jobs=args.n_jobs)(
        delayed(compute_losses_across_subj)(
            subj, subj_ids, args.pred_dir, args.target_dir, args.batch_size
        )
        for subj in tqdm(subj_ids)
    )
    print(f"Within MSE: {np.mean(within_mse)}, Std: {np.std(within_mse)}")
    print(f"Across MSE: {np.mean(across_mse)}, Std: {np.std(across_mse)}")
    np.save("within_and_across_mse.npy", {"within": within_mse, "across": across_mse})


if __name__ == "__main__":
    args = parser.parse_args()
    np.random.seed(args.seed)
    main(args)
