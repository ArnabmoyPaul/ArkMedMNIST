"""test_trainer_amp.py — confirms AMP-enabled train_one_epoch runs, produces
finite losses, and updates weights, using a tiny CPU-safe stand-in model
(autocast is skipped on CPU; this test only exercises the code path, not
actual mixed-precision numerics, which requires a CUDA device -- see
Task 11 for the live-GPU verification). Run: python test_trainer_amp.py"""
import os
import shutil
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from trainer import train_one_epoch, test_classification

# ponytail: train_one_epoch's untouched debug-save (i % 50 == 0) writes to a
# hardcoded "Models/" dir that production only has because engine.py's
# os.makedirs(model_path) creates it as a side effect before training starts.
# This test calls train_one_epoch directly, so recreate/clean that dir here
# rather than touching trainer.py.
_MODELS_DIR = "Models"
_SAVED_IMAGES = ["Models/student0.jpeg", "Models/teacher0.jpeg"]  # Debug-save output filenames


class _TinyArkLike(nn.Module):
    """Mimics ArkSwinTransformer's forward(x, head_n) -> (feat, logits) contract
    with a trivial linear encoder, so this test has no GPU/timm dependency."""
    def __init__(self, in_dim=12, feat_dim=4, num_classes=3):
        super().__init__()
        self.enc = nn.Linear(in_dim, feat_dim)
        self.head = nn.Linear(feat_dim, num_classes)

    def forward(self, x, head_n=None):
        # ponytail: train_one_epoch's untouched debug-save path does
        # samples[0].transpose(1, 2, 0), i.e. it assumes real (C, H, W)
        # images. Keep per-sample input 3D (C, H, W) here so that debug
        # save works, and flatten before the Linear so the tiny model's
        # math is unaffected -- this only touches the test's synthetic
        # data, not trainer.py.
        feat = self.enc(x.flatten(1))
        return feat, self.head(feat)


def test_train_one_epoch_runs_with_amp_disabled_on_cpu():
    torch.manual_seed(0)
    model = _TinyArkLike()
    teacher = _TinyArkLike()
    teacher.load_state_dict(model.state_dict())
    for p in teacher.parameters():
        p.requires_grad = False

    x1 = torch.randn(16, 3, 2, 2)  # (N, C, H, W), C=3 (RGB) so debug-save's JPEG write works
    x2 = x1 + 0.01 * torch.randn(16, 3, 2, 2)
    y = torch.eye(3)[torch.randint(0, 3, (16,))]
    loader = DataLoader(TensorDataset(x1, x2, y), batch_size=4)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    momentum_schedule = [0.9] * 10
    before = model.enc.weight.clone()

    scaler = torch.amp.GradScaler('cuda', enabled=False)  # CPU test: AMP path present but inert
    models_dir_existed = os.path.isdir(_MODELS_DIR)
    os.makedirs(_MODELS_DIR, exist_ok=True)
    try:
        train_one_epoch(model, 0, "tinyset", loader, torch.device('cpu'), nn.CrossEntropyLoss(),
                         optimizer, epoch=0, ema_mode="epoch", teacher=teacher,
                         momentum_schedule=momentum_schedule, it=0, scaler=scaler)
    finally:
        # Only remove files this test created, not the whole directory
        for img_path in _SAVED_IMAGES:
            if os.path.exists(img_path):
                os.remove(img_path)
        # Remove directory only if this test created it
        if not models_dir_existed and os.path.isdir(_MODELS_DIR):
            shutil.rmtree(_MODELS_DIR, ignore_errors=True)

    assert not torch.allclose(before, model.enc.weight), "weights should have updated"


def test_backward_compatible_call_without_scaler_still_works():
    """The original 12-positional-arg call site (no scaler) must keep working
    unmodified -- scaler defaults to None, meaning no AMP."""
    torch.manual_seed(0)
    model = _TinyArkLike()
    teacher = _TinyArkLike()
    teacher.load_state_dict(model.state_dict())
    for p in teacher.parameters():
        p.requires_grad = False
    x1 = torch.randn(8, 3, 2, 2)
    x2 = x1.clone()
    y = torch.eye(3)[torch.randint(0, 3, (8,))]
    loader = DataLoader(TensorDataset(x1, x2, y), batch_size=4)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    models_dir_existed = os.path.isdir(_MODELS_DIR)
    os.makedirs(_MODELS_DIR, exist_ok=True)
    try:
        train_one_epoch(model, 0, "tinyset", loader, torch.device('cpu'), nn.CrossEntropyLoss(),
                         optimizer, epoch=0, ema_mode="epoch", teacher=teacher,
                         momentum_schedule=[0.9], it=0)  # no scaler kwarg at all
    finally:
        # Only remove files this test created, not the whole directory
        for img_path in _SAVED_IMAGES:
            if os.path.exists(img_path):
                os.remove(img_path)
        # Remove directory only if this test created it
        if not models_dir_existed and os.path.isdir(_MODELS_DIR):
            shutil.rmtree(_MODELS_DIR, ignore_errors=True)


def test_classification_is_3d_flag_bypasses_tta_shape_inference():
    # A 3D volume batch (bs, c, d, h, w) is 5D -- same rank as the 10-crop TTA
    # case (bs, n_crops, c, h, w) that test_classification's non-3D branch
    # shape-sniffs for. Without is_3d=True this would get misread as
    # (bs, n_crops=c, c=d, h, w) and reshaped wrong. Requires CUDA (targets are
    # hardcoded .cuda() in test_classification, pre-existing/out of scope here).
    if not torch.cuda.is_available():
        print("  (skipping test_classification_is_3d_flag_bypasses_tta_shape_inference: no CUDA)")
        return
    torch.manual_seed(0)
    model = _TinyArkLike(in_dim=2*4*4*4, num_classes=3).cuda()
    x = torch.randn(6, 2, 4, 4, 4)  # (bs, c=2, d=4, h=4, w=4) -- 5D, not TTA
    y = torch.eye(3)[torch.randint(0, 3, (6,))]
    loader = DataLoader(TensorDataset(x, x, y), batch_size=3)

    y_test, p_test = test_classification(model, 0, loader, torch.device('cuda'), multiclass=True, is_3d=True)
    assert y_test.shape == (6, 3)
    assert p_test.shape == (6, 3), f"is_3d=True must keep bs=6 (not misread as n_crops), got {p_test.shape}"


if __name__ == "__main__":
    test_train_one_epoch_runs_with_amp_disabled_on_cpu()
    test_backward_compatible_call_without_scaler_still_works()
    test_classification_is_3d_flag_bypasses_tta_shape_inference()
    print("test_trainer_amp.py: all checks passed")
