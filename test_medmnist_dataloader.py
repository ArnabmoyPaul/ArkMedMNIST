"""test_medmnist_dataloader.py — self-check for medmnist_dataloader.py.
Run: python test_medmnist_dataloader.py"""
import numpy as np
from medmnist_dataloader import (
    MEDMNIST_2D_KEYS, medmnist_task_type, medmnist_num_classes,
    _to_onehot_or_multihot,
)

# Ground truth transcribed directly from the installed medmnist==3.0.2's
# medmnist.INFO during planning (see Task 4 for the full verified dump) --
# specifically covers the regressions the brief calls out: BreastMNIST is
# 2-class (the current file says 3), ChestMNIST is 14-way multi-label.
EXPECTED = {
    'breastmnist':   (2, 'multi-class classification'),
    'pneumoniamnist':(2, 'multi-class classification'),
    'chestmnist':    (14, 'multi-label classification'),
    'retinamnist':   (5, 'multi-class classification'),  # ordinal-regression -> CE, matches RSNAPneumonia's one-hot convention
    'organamnist':   (11, 'multi-class classification'),
}


def test_known_regressions():
    for key, (n_classes, task) in EXPECTED.items():
        assert medmnist_num_classes(key) == n_classes, \
            f"{key}: expected {n_classes} classes, got {medmnist_num_classes(key)}"
        assert medmnist_task_type(key) == task, \
            f"{key}: expected task={task}, got {medmnist_task_type(key)}"


def test_all_12_keys_resolve():
    assert len(MEDMNIST_2D_KEYS) == 12
    for key in MEDMNIST_2D_KEYS:
        assert medmnist_num_classes(key) > 0
        assert medmnist_task_type(key) in ('multi-class classification', 'multi-label classification')


def test_onehot_encoding_for_multiclass():
    # RSNAPneumonia's convention (dataloader.py): one-hot float vector, not a
    # bare class index -- trainer.py does `targets.float()` unconditionally,
    # so a bare long index would silently mismatch CrossEntropyLoss's shape.
    label = _to_onehot_or_multihot(raw_label=2, key='pathmnist', num_classes=9)
    assert label.dtype == np.float32
    assert label.shape == (9,)
    assert label.sum() == 1.0 and label[2] == 1.0


def test_multihot_passthrough_for_multilabel():
    raw = np.array([0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
    label = _to_onehot_or_multihot(raw_label=raw, key='chestmnist', num_classes=14)
    assert label.dtype == np.float32
    assert label.shape == (14,)
    assert np.array_equal(label, raw.astype(np.float32))


if __name__ == "__main__":
    test_known_regressions()
    test_all_12_keys_resolve()
    test_onehot_encoding_for_multiclass()
    test_multihot_passthrough_for_multilabel()
    print("test_medmnist_dataloader.py: all checks passed")
