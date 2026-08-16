"""test_medmnist3d_dataloader.py -- self-check for medmnist3d_dataloader.py.
Run: python test_medmnist3d_dataloader.py"""
import numpy as np
from medmnist_dataloader import _to_onehot_or_multihot
from medmnist3d_dataloader import DATASET_MAP_3D, NUM_CLASSES_MAP_3D, DATASETS_CONFIG_3D


def test_datasets_config_3d_matches_num_classes_map():
    assert set(DATASETS_CONFIG_3D) == set(DATASET_MAP_3D)
    for name, n_classes in NUM_CLASSES_MAP_3D.items():
        entry = DATASETS_CONFIG_3D[name]
        assert len(entry['diseases']) == n_classes, \
            f"{name}: expected {n_classes} disease names, got {len(entry['diseases'])}"
        assert entry['task_type'] == 'multi-class classification'


def test_onehot_encoding_for_3d_label():
    # MedMNIST3DWrapper.__getitem__ reuses medmnist_dataloader's
    # _to_onehot_or_multihot (not a bare torch.long class index) -- trainer.py
    # does targets.float() unconditionally, so a bare index would silently
    # mismatch CrossEntropyLoss's (B, num_classes) expectation. Verify the
    # reuse works with a 3D key (lowercased, matching MedMNIST3DWrapper's
    # `self.dataset_name.lower()` call).
    label = _to_onehot_or_multihot(raw_label=np.int64(2), key='fracturemnist3d', num_classes=3)
    assert label.dtype == np.float32
    assert label.shape == (3,)
    assert label.sum() == 1.0 and label[2] == 1.0

    label = _to_onehot_or_multihot(raw_label=np.int64(1), key='organmnist3d', num_classes=11)
    assert label.shape == (11,)
    assert label.sum() == 1.0 and label[1] == 1.0


if __name__ == "__main__":
    test_datasets_config_3d_matches_num_classes_map()
    test_onehot_encoding_for_3d_label()
    print("test_medmnist3d_dataloader.py: all checks passed")
