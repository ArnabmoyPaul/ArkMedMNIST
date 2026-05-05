
import torch
from torch.utils.data import Dataset
import medmnist
import numpy as np


# ─── Registry of all 6 MedMNIST 3D datasets ──────────────────────────────────
DATASET_MAP_3D = {
    'OrganMNIST3D':    medmnist.OrganMNIST3D,
    'NoduleMNIST3D':   medmnist.NoduleMNIST3D,
    'AdrenalMNIST3D':  medmnist.AdrenalMNIST3D,
    'FractureMNIST3D': medmnist.FractureMNIST3D,
    'VesselMNIST3D':   medmnist.VesselMNIST3D,
    'SynapseMNIST3D':  medmnist.SynapseMNIST3D,
}

NUM_CLASSES_MAP_3D = {
    'OrganMNIST3D':    11,
    'NoduleMNIST3D':    2,
    'AdrenalMNIST3D':   2,
    'FractureMNIST3D':  3,
    'VesselMNIST3D':    2,
    'SynapseMNIST3D':   2,
}

TASK_MAP_3D = {name: 'multi-class classification' for name in DATASET_MAP_3D}


class Augment3D:
    """Stochastic 3D augmentation for teacher-student dual-view training."""

    def __init__(self, training=True):
        self.training = training

    def __call__(self, vol):
        """vol: np.float32 (D,H,W) in [0,1]  →  torch (1,D,H,W)"""
        if self.training:
            for ax in range(3):
                if np.random.rand() > 0.5:
                    vol = np.flip(vol, ax).copy()
            scale = np.random.uniform(0.85, 1.15)
            shift = np.random.uniform(-0.05, 0.05)
            vol = np.clip(vol * scale + shift, 0.0, 1.0)
        return torch.tensor(vol[None], dtype=torch.float32)


class MedMNIST3DWrapper(Dataset):
    """
    Wraps any MedMNIST 3D dataset.
    Returns (view1, view2, label) — dual-view for teacher-student training.
    """

    def __init__(self, dataset_name, split='train'):
        assert dataset_name in DATASET_MAP_3D, (
            f"Unknown 3D dataset: '{dataset_name}'. "
            f"Choose from {list(DATASET_MAP_3D)}"
        )
        self.dataset_name = dataset_name
        self.num_classes  = NUM_CLASSES_MAP_3D[dataset_name]
        self.task_type    = TASK_MAP_3D[dataset_name]

        raw = DATASET_MAP_3D[dataset_name](split=split, download=True)
        self.imgs   = raw.imgs.astype(np.float32) / 255.0  # (N,28,28,28) in [0,1]
        self.labels = raw.labels.squeeze().astype(np.int64)  # (N,)
        self.aug = Augment3D(training=(split == 'train'))

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        vol   = self.imgs[idx]           # (28,28,28)
        label = int(self.labels[idx])
        v1    = self.aug(vol)            # student view
        v2    = self.aug(vol)            # teacher view
        return v1, v2, torch.tensor(label, dtype=torch.long)
