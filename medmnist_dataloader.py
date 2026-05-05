
import torch
from torch.utils.data import Dataset
import medmnist
import numpy as np
from PIL import Image
import torchvision.transforms as transforms


def get_transform(size=64, mode='train'):
    if mode == 'train':
        return transforms.Compose([
            transforms.Resize((size, size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])


DATASET_MAP = {
    'PathMNIST':      medmnist.PathMNIST,
    'BloodMNIST':     medmnist.BloodMNIST,
    'DermaMNIST':     medmnist.DermaMNIST,
    'OCTMNIST':       medmnist.OCTMNIST,
    'PneumoniaMNIST': medmnist.PneumoniaMNIST,
    'RetinaMNIST':    medmnist.RetinaMNIST,
    'BreastMNIST':    medmnist.BreastMNIST,
    'TissueMNIST':    medmnist.TissueMNIST,
    'OrganAMNIST':    medmnist.OrganAMNIST,
    'OrganCMNIST':    medmnist.OrganCMNIST,
    'OrganSMNIST':    medmnist.OrganSMNIST,
    'ChestMNIST':     medmnist.ChestMNIST,
}

NUM_CLASSES_MAP = {
    'PathMNIST':      9,
    'BloodMNIST':     8,
    'DermaMNIST':     7,
    'OCTMNIST':       4,
    'PneumoniaMNIST': 2,
    'RetinaMNIST':    5,
    'BreastMNIST':    3,
    'TissueMNIST':    8,
    'OrganAMNIST':    11,
    'OrganCMNIST':    11,
    'OrganSMNIST':    11,
    'ChestMNIST':     14,
}

TASK_MAP = {
    'PathMNIST':      'multi-class classification',
    'BloodMNIST':     'multi-class classification',
    'DermaMNIST':     'multi-class classification',
    'OCTMNIST':       'multi-class classification',
    'PneumoniaMNIST': 'multi-class classification',
    'RetinaMNIST':    'multi-class classification',
    'BreastMNIST':    'multi-class classification',
    'TissueMNIST':    'multi-class classification',
    'OrganAMNIST':    'multi-class classification',
    'OrganCMNIST':    'multi-class classification',
    'OrganSMNIST':    'multi-class classification',
    'ChestMNIST':     'multi-label classification',
}


class MedMNISTWrapper(Dataset):
    def __init__(self, dataset_name, split='train', size=64):
        assert dataset_name in DATASET_MAP, \
            f"{dataset_name} not found"

        self.dataset_name = dataset_name
        self.split        = split
        self.size         = size
        self.num_classes  = NUM_CLASSES_MAP[dataset_name]
        self.task_type    = TASK_MAP[dataset_name]

        self.data = DATASET_MAP[dataset_name](
            split=split,
            download=True,
            size=size,
            as_rgb=True
        )

        self.train_transform = get_transform(size, mode='train')
        self.val_transform   = get_transform(size, mode='val')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, label = self.data[idx]

        if not isinstance(img, Image.Image):
            img = Image.fromarray(np.uint8(img))

        label = torch.tensor(label).squeeze()

        if self.task_type == 'multi-class classification':
            label = label.long()
        else:
            label = label.float()

        if self.split == 'train':
            view1 = self.train_transform(img)
            view2 = self.train_transform(img)
        else:
            view1 = self.val_transform(img)
            view2 = self.val_transform(img)

        return view1, view2, label
