"""MedMNIST Dataset classes matching Ark_Plus/Pretraining's dataloader.py
Dataset contract exactly: __init__(images_path, file_path, crop_size, resize,
augment, num_class, annotation_percent) and __getitem__ -> (student_img,
teacher_img, label) as CHW float32 tensors -- so main_ark.py's/engine.py's/
trainer.py's unmodified construction and training code works completely
unchanged (Phase 0: add a data layer against their existing interfaces,
don't rewrite the loop).

`file_path` is repurposed to carry the medmnist split name ('train'/'val'/
'test') instead of an annotation file path -- MedMNIST downloads by
(dataset key, split), it has no annotation file. `images_path` is unused
(kept only for signature parity with dict_dataloarder's other entries).
"""
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import medmnist
from medmnist import INFO

MEDMNIST_2D_KEYS = [
    'pathmnist', 'bloodmnist', 'dermamnist', 'octmnist', 'pneumoniamnist',
    'retinamnist', 'breastmnist', 'tissuemnist', 'organamnist',
    'organcmnist', 'organsmnist', 'chestmnist',
]


def medmnist_task_type(key):
    """'multi-label classification' | 'multi-class classification', matching
    datasets_config.yaml's exact task_type vocabulary (engine.py branches on
    `== "multi-class classification"` verbatim). binary-class and
    ordinal-regression both bucket to multi-class/CE, matching how
    RSNAPneumonia (3-way) and Shenzhen-style binary tasks are already typed
    in the untouched datasets_config.yaml."""
    return 'multi-label classification' if INFO[key]['task'].startswith('multi-label') \
        else 'multi-class classification'


def medmnist_num_classes(key):
    return len(INFO[key]['label'])


def _to_onehot_or_multihot(raw_label, key, num_classes):
    """RSNAPneumonia (dataloader.py) feeds CrossEntropyLoss a one-hot float
    vector, not a bare class index, because trainer.py does `targets.float()`
    unconditionally on every batch (train_one_epoch and evaluate both) --
    a bare long class index would silently produce the wrong shape/dtype for
    CrossEntropyLoss against (B, num_classes) logits. Multi-label targets are
    already multi-hot from medmnist and just need a float cast."""
    raw = np.asarray(raw_label).squeeze()
    if medmnist_task_type(key) == 'multi-label classification':
        return raw.astype(np.float32)
    onehot = np.zeros(num_classes, dtype=np.float32)
    onehot[int(raw)] = 1.0
    return onehot


def _student_teacher_transforms(crop_size, resize):
    """Mirrors ChestXray14.__getitem__'s augment=None branch (dataloader.py):
    teacher gets a deterministic resize-only view, student gets randomized
    crop/rotation/color-jitter -- Ark+'s asymmetric-input design (stable
    teacher signal). Native MedMNIST images are 28x28; resize/crop still
    apply meaningfully to reach crop_size."""
    teacher_tf = transforms.Compose([
        transforms.Resize((crop_size, crop_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    student_tf = transforms.Compose([
        transforms.Resize((resize, resize)),
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return student_tf, teacher_tf


class MedMNIST2DDataset(Dataset):
    def __init__(self, images_path, file_path, crop_size=224, resize=256, augment=None,
                 num_class=None, annotation_percent=100, key=None):
        assert key in MEDMNIST_2D_KEYS, f"unknown MedMNIST 2D key: {key}"
        self.key = key
        split = file_path  # repurposed: 'train' | 'val' | 'test'
        assert split in ('train', 'val', 'test'), \
            f"MedMNIST2DDataset expects file_path to be a split name, got {file_path!r}"
        cls = getattr(medmnist, INFO[key]['python_class'])
        self.data = cls(split=split, download=True, size=28, as_rgb=True)
        self.num_classes = medmnist_num_classes(key)
        self.is_train = (split == 'train')
        self.augment = augment  # deterministic transform for val/test (see main_ark.py)

        if self.is_train:
            self.student_tf, self.teacher_tf = _student_teacher_transforms(crop_size, resize)
        else:
            assert augment is not None, \
                "val/test MedMNIST2DDataset requires an `augment` transform, matching " \
                "ChestXray14's convention (main_ark.py passes build_transform_classification(...))"

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, raw_label = self.data[idx]
        if not isinstance(img, Image.Image):
            img = Image.fromarray(np.uint8(img))
        label = torch.from_numpy(_to_onehot_or_multihot(raw_label, self.key, self.num_classes))
        if self.is_train:
            return self.student_tf(img), self.teacher_tf(img), label
        return self.augment(img), self.augment(img), label


def _make_medmnist_dataset_class(key):
    """dict_dataloarder needs one no-extra-arg class per dataset -- main_ark.py
    calls dict_dataloarder[dataset](images_path=..., file_path=..., ...) with
    no way to pass `key` through, so bind it via a tiny subclass per dataset."""
    class _Bound(MedMNIST2DDataset):
        def __init__(self, images_path, file_path, crop_size=224, resize=256,
                     augment=None, num_class=None, annotation_percent=100):
            super().__init__(images_path, file_path, crop_size, resize, augment,
                              num_class, annotation_percent, key=key)
    _Bound.__name__ = f"MedMNIST_{key}"
    return _Bound


MEDMNIST_DATALOADER_DICT = {key: _make_medmnist_dataset_class(key) for key in MEDMNIST_2D_KEYS}
