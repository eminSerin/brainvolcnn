import os.path as op

import numpy as np
import torch
from torch.utils.data import Dataset

from .utils import _unmask_timeseries


class TaskGenDataset(Dataset):
    """Dataset class for task generation experiment.

    This class can be used for CNN models.

    Parameters
    ----------
    subj_ids : np.ndarray
        List of subject IDs.
    rest_dir : str
        Path to directory containing resting state timeseries.
    task_dir : str
        Path to directory containing task timeseries.
    num_samples : int, optional
        Number of samples per subject, by default 8
    device : str, optional
        Device to load data to, by default "cpu"
    unmask : bool, optional
        Whether or not to unmask timeseries, by default None.
    mask : nib.Nifti1Image, optional
        Mask array, by default None
    crop : bool, optional
        Crops the image to get rid of unnecessary blank spaces around the borders of brain, by default False.

    Returns
    ----------
    torch.utils.data.Dataset:
        Dataset object for task generation experiment.
    """

    def __init__(
        self,
        subj_ids,
        rest_dir,
        task_dir,
        num_samples=8,
        device="cpu",
        unmask=None,
        mask=None,
        crop=False,
    ) -> None:
        super().__init__()
        self.subj_ids = subj_ids
        self.rest_dir = rest_dir
        self.task_dir = task_dir
        self.num_samples = num_samples
        self.device = device
        self.unmask = unmask
        self.mask = mask
        self.crop = crop

        if (unmask is not None) and (mask is None):
            raise ValueError("Mask must be provided to unmask timeseries!")

    def __getitem__(self, idx):
        subj = self.subj_ids[idx]
        sample_id = np.random.randint(0, self.num_samples)
        # Rest
        rest_file = op.join(self.rest_dir, f"{subj}_sample{sample_id}_rsfc.npy")
        if not op.exists(rest_file):
            raise FileExistsError(f"{rest_file} does not exist!")
        # Task
        task_file = op.join(self.task_dir, f"{subj}_joint_MNI_task_contrasts.npy")
        if not op.exists(task_file):
            raise FileExistsError(f"{task_file} does not exist!")

        if self.unmask is not None:
            return torch.from_numpy(
                _unmask_timeseries(np.load(rest_file).T, self.mask, self.crop)
            ).type(torch.float).to(self.device), torch.from_numpy(
                _unmask_timeseries(np.load(task_file), self.mask, self.crop)
            ).type(
                torch.float
            ).to(
                self.device
            )
        else:
            return torch.from_numpy(np.load(rest_file).T).type(torch.float).to(
                self.device
            ), torch.from_numpy(np.load(task_file).T).type(torch.float).to(self.device)

    def __len__(self):
        return len(self.subj_ids)


class TaskGenDatasetLinear(TaskGenDataset):
    """Dataset class for task generation experiment using simple linear regression method presented in Tavor et al., 2016."""

    def __init__(
        self,
        subj_ids,
        rest_dir,
        task_dir,
        num_samples=8,
        device="cpu",
        unmask=None,
        mask=None,
        crop=False,
    ) -> None:
        super().__init__(
            subj_ids, rest_dir, task_dir, num_samples, device, unmask, mask, crop
        )

    def __getitem__(self, idx):
        rest_data, task_data = super().__getitem__(idx)
        return rest_data.flatten(1), task_data.flatten(1)
