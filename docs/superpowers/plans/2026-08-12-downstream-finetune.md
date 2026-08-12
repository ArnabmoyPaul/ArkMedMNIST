# Downstream Fine-Tuning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `finetune_downstream.py`, a driver that loads the epoch-20 checkpoint's teacher weights into a fresh single-head model and full-fine-tunes it on one MedMNIST-2D dataset at a time, reporting test AUC.

**Architecture:** Reuse `build_omni_model`, `dict_dataloarder`, `build_transform_classification`, `trainer.evaluate`, `trainer.test_classification`, and `metric_AUROC` unchanged. Add one new plain-supervised training-step function (`trainer.train_downstream_epoch`, no EMA/distillation — that's what `train_one_epoch` is for) and one new checkpoint-loading helper (`models.load_finetune_backbone`, strips `omni_heads.*` before loading so the wrong dataset's head can never get spliced in). The driver script wires these together, tracks val loss for best-checkpoint selection and early stopping, and reports final test AUC.

**Tech Stack:** PyTorch, timm (`ArkSwinTransformer` only — no `create_optimizer`/`create_scheduler`, see Task 3 rationale), existing `medmnist_dataloader.py` / `dataloader.py` dataset plumbing.

## Global Constraints

- One script, run once per dataset: `python finetune_downstream.py --dataset <key>` (12 independent invocations — no in-process loop over all 12).
- Initialize from the checkpoint's **teacher** weights, **full fine-tune** (no linear-probe mode).
- No crash-proof resume — a single-dataset fine-tune is cheap enough to just rerun.
- Default `--checkpoint` = `Models/swin_tiny_medmnist2d_swintiny/Ark_Plus_pathmnist_bloodmnist_dermamnist_octmnist_pneumoniamnist_retinamnist_breastmnist_tissuemnist_organamnist_organcmnist_organsmnist_chestmnist/medmnist2d_swintiny/Ark_Plus_pathmnist_bloodmnist_dermamnist_octmnist_pneumoniamnist_retinamnist_breastmnist_tissuemnist_organamnist_organcmnist_organsmnist_chestmnist20.pth.tar` (epoch 20: student mAUC 0.8982, teacher mAUC 0.9100 — both peaks of the pretraining run).
- Defaults: `--lr 1e-3`, `--epochs 20`, `--patience 5`, `--batch_size 32`, `crop_size=112`/`resize=128` (same as pretraining, hardcoded not CLI-exposed — no requirement to vary them).
- Output layout: checkpoint → `Models/finetune_<dataset>/best.pth.tar`; results → `Outputs/finetune_<dataset>/results.json` (single summary dict) and `results.txt` (human-readable, appended, matching `engine.py`'s existing report format).
- Test file naming/style: plain `test_*` functions, no pytest fixtures/classes, `if __name__ == "__main__":` block running each test and printing `"<file>: all checks passed"` — matches every existing test file in this repo (`test_trainer_amp.py`, `test_models_medmnist.py`, `test_engine_resume_indexing.py`).

---

## File Structure

- Modify: `trainer.py` — add `train_downstream_epoch()`
- Create: `test_trainer_downstream.py` — test for `train_downstream_epoch()`
- Modify: `models.py` — add `load_finetune_backbone()`
- Modify: `test_models_medmnist.py` — add tests for `load_finetune_backbone()`
- Create: `finetune_downstream.py` — CLI driver
- Create: `test_finetune_downstream.py` — tests for the driver's pure logic (`_early_stop`, `parse_args`)

---

### Task 1: `train_downstream_epoch` in trainer.py

**Files:**
- Modify: `trainer.py`
- Test: `test_trainer_downstream.py`

**Interfaces:**
- Produces: `train_downstream_epoch(model, use_head_n, dataset, data_loader_train, device, criterion, optimizer, epoch, scaler=None) -> float` (returns average training loss for the epoch). Dataloader batches are `(samples, _, targets)` triples — the second (teacher-view) element is ignored, exactly like `evaluate()` already does in the same file.

- [ ] **Step 1: Write the failing test**

Create `test_trainer_downstream.py`:

```python
"""test_trainer_downstream.py — Run: python test_trainer_downstream.py"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from trainer import train_downstream_epoch


class _TinyArkLike(nn.Module):
    """Mimics ArkSwinTransformer's forward(x, head_n) -> (feat, logits) contract
    with a trivial linear encoder, so this test has no GPU/timm dependency."""
    def __init__(self, in_dim=12, feat_dim=4, num_classes=3):
        super().__init__()
        self.enc = nn.Linear(in_dim, feat_dim)
        self.head = nn.Linear(feat_dim, num_classes)

    def forward(self, x, head_n=None):
        feat = self.enc(x.flatten(1))
        return feat, self.head(feat)


def test_train_downstream_epoch_updates_weights_and_returns_finite_loss():
    torch.manual_seed(0)
    model = _TinyArkLike()
    x1 = torch.randn(16, 3, 2, 2)
    x2 = torch.randn(16, 3, 2, 2)  # second (teacher) view -- must be ignored
    y = torch.eye(3)[torch.randint(0, 3, (16,))]
    loader = DataLoader(TensorDataset(x1, x2, y), batch_size=4)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    before = model.enc.weight.clone()

    avg_loss = train_downstream_epoch(model, 0, "tinyset", loader, torch.device('cpu'),
                                       nn.CrossEntropyLoss(), optimizer, epoch=0, scaler=None)

    assert not torch.allclose(before, model.enc.weight), "weights should have updated"
    assert avg_loss == avg_loss and avg_loss > 0, f"expected a finite positive loss, got {avg_loss}"


def test_train_downstream_epoch_ignores_second_view():
    # If the function accidentally trained on x2 instead of x1, feeding x2 as
    # garbage (huge magnitude) while x1 stays well-scaled must not blow up the loss.
    torch.manual_seed(0)
    model = _TinyArkLike()
    x1 = torch.randn(16, 3, 2, 2)
    x2 = torch.randn(16, 3, 2, 2) * 1000
    y = torch.eye(3)[torch.randint(0, 3, (16,))]
    loader = DataLoader(TensorDataset(x1, x2, y), batch_size=4)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    avg_loss = train_downstream_epoch(model, 0, "tinyset", loader, torch.device('cpu'),
                                       nn.CrossEntropyLoss(), optimizer, epoch=0, scaler=None)
    assert avg_loss < 100, f"loss exploded ({avg_loss}) -- second view leaked into training"


if __name__ == "__main__":
    test_train_downstream_epoch_updates_weights_and_returns_finite_loss()
    test_train_downstream_epoch_ignores_second_view()
    print("test_trainer_downstream.py: all checks passed")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python test_trainer_downstream.py`
Expected: `ImportError: cannot import name 'train_downstream_epoch' from 'trainer'`

- [ ] **Step 3: Implement `train_downstream_epoch` in trainer.py**

Add to `trainer.py`, after `train_one_epoch` (before `ema_update_teacher`):

```python
def train_downstream_epoch(model, use_head_n, dataset, data_loader_train, device, criterion, optimizer, epoch, scaler=None):
    """Plain supervised fine-tuning step -- no teacher, no EMA, no distillation
    term. train_one_epoch above is hard-wired to the omni self-distillation
    loss (teacher forward pass + momentum-scheduled MSE consistency term);
    none of that applies to downstream fine-tuning on a single labeled task."""
    batch_time = MetricLogger('Time', ':6.3f')
    losses = MetricLogger('Loss_'+dataset, ':.4e')
    progress = ProgressLogger(
        len(data_loader_train),
        [batch_time, losses],
        prefix="Epoch: [{}]".format(epoch))

    model.train()
    amp_enabled = scaler is not None and scaler.is_enabled()
    end = time.time()
    for i, (samples, _, targets) in enumerate(data_loader_train):
        samples, targets = samples.float().to(device), targets.float().to(device)

        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
            _, pred = model(samples, use_head_n)
            loss = criterion(pred, targets)

        optimizer.zero_grad()
        if amp_enabled:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        losses.update(loss.item(), samples.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 50 == 0:
            progress.display(i)

    print(f"  {dataset}: loss={losses.avg:.4f}")
    return losses.avg
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python test_trainer_downstream.py`
Expected: `test_trainer_downstream.py: all checks passed`

- [ ] **Step 5: Commit**

```bash
git add trainer.py test_trainer_downstream.py
git commit -m "add train_downstream_epoch: plain supervised fine-tuning step, no distillation"
```

---

### Task 2: `load_finetune_backbone` in models.py

**Files:**
- Modify: `models.py`
- Modify: `test_models_medmnist.py`

**Interfaces:**
- Consumes: `ArkSwinTransformer`, `build_omni_model` (already in `models.py`, unchanged).
- Produces: `load_finetune_backbone(model, checkpoint_path, key='teacher') -> torch.nn.modules.module._IncompatibleKeys` (loads encoder weights in place, prints the load message, returns it — mirrors `load_imagenet_backbone`'s contract).

- [ ] **Step 1: Write the failing test**

Add to `test_models_medmnist.py` (add `import os` to the existing import block at the top, then append these two tests before the `if __name__ == "__main__":` block):

```python
def test_load_finetune_backbone_transfers_encoder_not_heads():
    # Build a "pretrained" 3-head model (mimicking the 12-dataset omni checkpoint,
    # scaled down) and save its teacher state dict as a fake checkpoint.
    pretrained = ArkSwinTransformer([9, 2, 5], projector_features=None, use_mlp=False,
                                     patch_size=4, window_size=7, embed_dim=96,
                                     depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24))
    ckpt_path = "test_fake_checkpoint_heads.pth.tar"
    torch.save({"teacher": pretrained.state_dict()}, ckpt_path)

    try:
        # Target's single head happens to match pretrained head index 1's size (2
        # classes) -- must NOT silently pick up head 1's weights by name collision.
        target = ArkSwinTransformer([2], projector_features=None, use_mlp=False,
                                     patch_size=4, window_size=7, embed_dim=96,
                                     depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24))
        head_before = target.omni_heads[0].weight.clone()
        load_finetune_backbone(target, ckpt_path, key="teacher")

        assert torch.allclose(target.omni_heads[0].weight, head_before), \
            "omni_heads must stay untouched -- loading them risks splicing in the wrong dataset's head"
        assert torch.allclose(target.patch_embed.proj.weight, pretrained.patch_embed.proj.weight), \
            "encoder weights must transfer from the checkpoint"
    finally:
        os.remove(ckpt_path)


def test_load_finetune_backbone_survives_class_count_matching_no_head():
    pretrained = ArkSwinTransformer([9, 2, 5], projector_features=None, use_mlp=False,
                                     patch_size=4, window_size=7, embed_dim=96,
                                     depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24))
    ckpt_path = "test_fake_checkpoint_noheadmatch.pth.tar"
    torch.save({"teacher": pretrained.state_dict()}, ckpt_path)
    try:
        target = ArkSwinTransformer([7], projector_features=None, use_mlp=False,
                                     patch_size=4, window_size=7, embed_dim=96,
                                     depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24))
        load_finetune_backbone(target, ckpt_path, key="teacher")  # must not raise
    finally:
        os.remove(ckpt_path)
```

Also update the `if __name__ == "__main__":` block at the bottom to call both new tests, and the import line to include `load_finetune_backbone`:

```python
from models import ArkSwinTransformer, build_omni_model, load_imagenet_backbone, load_finetune_backbone
```

```python
if __name__ == "__main__":
    test_forward_features_pooling_bridges_timm_spatial_output()
    test_generate_embeddings_also_pools()
    test_build_omni_model_swin_tiny_at_custom_img_size()
    test_load_imagenet_backbone_transfers_encoder_not_heads()
    test_load_imagenet_backbone_at_112px_survives_window_shrink()
    test_load_finetune_backbone_transfers_encoder_not_heads()
    test_load_finetune_backbone_survives_class_count_matching_no_head()
    print("test_models_medmnist.py: all checks passed")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python test_models_medmnist.py`
Expected: `ImportError: cannot import name 'load_finetune_backbone' from 'models'`

- [ ] **Step 3: Implement `load_finetune_backbone` in models.py**

Add to `models.py`, after `load_imagenet_backbone`:

```python
def load_finetune_backbone(model, checkpoint_path, key='teacher'):
    """Loads a single-task-head model's encoder from a 12-head omni-pretraining
    checkpoint. omni_heads keys are always dropped before loading: the
    checkpoint's omni_heads.0 is whichever dataset was first in the ORIGINAL
    pretraining dataset_list (pathmnist), not this model's target dataset --
    loading it unfiltered would crash on a shape mismatch, or worse, silently
    splice in the wrong dataset's head whenever class counts happen to match
    (strict=False only tolerates missing/extra keys, not shape mismatches, so
    a genuine mismatch would still raise -- the head must be filtered out
    before load_state_dict ever sees it)."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint[key]
    state_dict = {k: v for k, v in state_dict.items() if not k.startswith('omni_heads.')}
    msg = model.load_state_dict(state_dict, strict=False)
    print('Loaded {} backbone from {} with msg: {}'.format(key, checkpoint_path, msg))
    return msg
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python test_models_medmnist.py`
Expected: `test_models_medmnist.py: all checks passed`

- [ ] **Step 5: Commit**

```bash
git add models.py test_models_medmnist.py
git commit -m "add load_finetune_backbone: transfer encoder from an omni checkpoint, strip heads"
```

---

### Task 3: `finetune_downstream.py` driver script

**Files:**
- Create: `finetune_downstream.py`
- Test: `test_finetune_downstream.py`

**Interfaces:**
- Consumes: `train_downstream_epoch` (Task 1), `load_finetune_backbone` (Task 2), `build_omni_model`/`save_checkpoint` (`models.py`, unchanged), `evaluate`/`test_classification` (`trainer.py`, unchanged), `dict_dataloarder`/`build_transform_classification` (`dataloader.py`, unchanged), `MEDMNIST_2D_KEYS`/`MEDMNIST_DATALOADER_DICT` (`medmnist_dataloader.py`, unchanged), `get_config`/`metric_AUROC` (`utils.py`, unchanged).
- Produces: `_early_stop(val_losses: list[float], patience: int) -> bool`, `parse_args(argv=None) -> argparse.Namespace`, `DEFAULT_CHECKPOINT: str`, `main(argv=None) -> dict` (the results summary dict, also written to `results.json`).

**Design note — why no `timm.create_optimizer`/`create_scheduler`:** `engine.py` builds its optimizer/scheduler through `main_ark.py`'s 40-option `get_args_parser()`, most of which (`ema_mode`, `momentum_teacher`, `pretrain_epochs`, `dataset_list`, ...) is omni-pretraining-specific and irrelevant here. This script only needs plain constant-LR momentum-SGD (fine-tuning runs are ≤20 epochs; a cosine schedule isn't part of the approved design), so it calls `torch.optim.SGD` directly instead of pulling in that whole parser just to satisfy `create_optimizer`'s expected attribute set. `build_omni_model` still needs *its* narrow set of fields (`model_name`, `projector_features`, `use_mlp`, `crop_size`, `pretrained_weights`), supplied via a small `SimpleNamespace` — same pattern `test_models_medmnist.py` already uses with `optparse.Values()`.

- [ ] **Step 1: Write the failing test**

Create `test_finetune_downstream.py`:

```python
"""test_finetune_downstream.py — Run: python test_finetune_downstream.py"""
from finetune_downstream import _early_stop, parse_args, DEFAULT_CHECKPOINT


def test_no_stop_while_still_improving():
    assert not _early_stop([0.9, 0.8, 0.7, 0.6, 0.5], patience=2)


def test_no_stop_before_patience_window_elapses():
    # best is index 0; only 2 epochs have passed since -- patience=3 not yet exceeded
    assert not _early_stop([0.5, 0.6, 0.7], patience=3)


def test_stops_after_patience_epochs_without_improvement():
    # best is index 1 (0.4); 3 epochs have passed since with no improvement
    assert _early_stop([0.9, 0.4, 0.5, 0.6, 0.7], patience=3)


def test_stops_exactly_at_patience_boundary():
    # best is index 0; exactly `patience` epochs have passed since (indices 1, 2)
    assert _early_stop([0.5, 0.6, 0.7], patience=2)


def test_default_checkpoint_points_at_epoch_20():
    assert DEFAULT_CHECKPOINT.endswith("20.pth.tar")
    assert "medmnist2d_swintiny" in DEFAULT_CHECKPOINT


def test_parse_args_defaults():
    args = parse_args(["--dataset", "chestmnist"])
    assert args.dataset == "chestmnist"
    assert args.checkpoint == DEFAULT_CHECKPOINT
    assert args.epochs == 20
    assert args.lr == 1e-3
    assert args.patience == 5


if __name__ == "__main__":
    test_no_stop_while_still_improving()
    test_no_stop_before_patience_window_elapses()
    test_stops_after_patience_epochs_without_improvement()
    test_stops_exactly_at_patience_boundary()
    test_default_checkpoint_points_at_epoch_20()
    test_parse_args_defaults()
    print("test_finetune_downstream.py: all checks passed")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python test_finetune_downstream.py`
Expected: `ModuleNotFoundError: No module named 'finetune_downstream'`

- [ ] **Step 3: Implement `finetune_downstream.py`**

Create `finetune_downstream.py`:

```python
import argparse
import json
import os
import time
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataloader import dict_dataloarder, build_transform_classification
from medmnist_dataloader import MEDMNIST_2D_KEYS, MEDMNIST_DATALOADER_DICT
from models import build_omni_model, load_finetune_backbone, save_checkpoint
from trainer import train_downstream_epoch, evaluate, test_classification
from utils import get_config, metric_AUROC

dict_dataloarder.update(MEDMNIST_DATALOADER_DICT)

CROP_SIZE, RESIZE = 112, 128  # match the pretraining run

DEFAULT_CHECKPOINT = os.path.join(
    "Models", "swin_tiny_medmnist2d_swintiny",
    "Ark_Plus_pathmnist_bloodmnist_dermamnist_octmnist_pneumoniamnist_retinamnist_"
    "breastmnist_tissuemnist_organamnist_organcmnist_organsmnist_chestmnist",
    "medmnist2d_swintiny",
    "Ark_Plus_pathmnist_bloodmnist_dermamnist_octmnist_pneumoniamnist_retinamnist_"
    "breastmnist_tissuemnist_organamnist_organcmnist_organsmnist_chestmnist20.pth.tar",
)


def _early_stop(val_losses, patience):
    """True once val_losses hasn't set a new best in the last `patience` epochs."""
    if len(val_losses) <= patience:
        return False
    best_idx = min(range(len(val_losses)), key=lambda i: val_losses[i])
    return best_idx <= len(val_losses) - 1 - patience


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Downstream fine-tune from an omni-pretrained checkpoint")
    p.add_argument("--dataset", required=True, choices=MEDMNIST_2D_KEYS)
    p.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--workers", type=int, default=4)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    datasets_config = get_config("datasets_config_medmnist.yaml")
    ds_cfg = datasets_config[args.dataset]
    diseases = ds_cfg["diseases"]
    num_classes = len(diseases)
    multiclass = ds_cfg["task_type"] == "multi-class classification"
    criterion = torch.nn.CrossEntropyLoss() if multiclass else torch.nn.BCEWithLogitsLoss()

    train_set = dict_dataloarder[args.dataset](images_path=ds_cfg["data_dir"], file_path=ds_cfg["train_list"],
                                                crop_size=CROP_SIZE, resize=RESIZE, augment=None)
    val_set = dict_dataloarder[args.dataset](images_path=ds_cfg["data_dir"], file_path=ds_cfg["val_list"],
                                              crop_size=CROP_SIZE, resize=RESIZE,
                                              augment=build_transform_classification(
                                                  normalize="imagenet", crop_size=CROP_SIZE, resize=RESIZE, mode="valid"))
    test_set = dict_dataloarder[args.dataset](images_path=ds_cfg["data_dir"], file_path=ds_cfg["test_list"],
                                               crop_size=CROP_SIZE, resize=RESIZE,
                                               augment=build_transform_classification(
                                                   normalize="imagenet", crop_size=CROP_SIZE, resize=RESIZE, mode="test"))

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=max(1, args.batch_size // 2), shuffle=False,
                              num_workers=args.workers, pin_memory=True)

    model_args = SimpleNamespace(model_name="swin_tiny", projector_features=512, use_mlp=False,
                                  crop_size=CROP_SIZE, pretrained_weights=None)
    model = build_omni_model(model_args, num_classes_list=[num_classes])
    load_finetune_backbone(model, args.checkpoint, key="teacher")
    model.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    model_dir = os.path.join("Models", f"finetune_{args.dataset}")
    output_dir = os.path.join("Outputs", f"finetune_{args.dataset}")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    best_path = os.path.join(model_dir, "best")

    val_losses = []
    best_val_loss = float("inf")
    for epoch in range(args.epochs):
        epoch_start = time.time()
        train_loss = train_downstream_epoch(model, 0, args.dataset, train_loader, device, criterion,
                                             optimizer, epoch, scaler=scaler)
        val_loss = evaluate(model, 0, val_loader, device, criterion, args.dataset, scaler=scaler)
        val_losses.append(val_loss)
        print(f"Epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
              f"({(time.time() - epoch_start) / 60:.1f} min)")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint({"epoch": epoch, "val_loss": val_loss, "state_dict": model.state_dict()},
                             filename=best_path)

        if _early_stop(val_losses, args.patience):
            print(f"Early stopping at epoch {epoch} (no val improvement in {args.patience} epochs)")
            break

    best_ckpt = torch.load(best_path + ".pth.tar", map_location=device)
    model.load_state_dict(best_ckpt["state_dict"])
    print(f"Loaded best checkpoint from epoch {best_ckpt['epoch']} (val_loss={best_ckpt['val_loss']:.4f})")

    y_test, p_test = test_classification(model, 0, test_loader, device, multiclass)
    individual_auc = metric_AUROC(y_test, p_test, len(diseases))
    overall_auc = float(np.mean(individual_auc))

    results = {
        "dataset": args.dataset,
        "checkpoint": args.checkpoint,
        "best_epoch": best_ckpt["epoch"],
        "best_val_loss": best_val_loss,
        "test_auc_per_class": dict(zip(diseases, [float(a) for a in individual_auc])),
        "test_overall_mAUC": overall_auc,
    }
    print(f"  Test AUC per class: {results['test_auc_per_class']}")
    print(f"  --- Overall test mAUC: {overall_auc:.4f} ---")

    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    with open(os.path.join(output_dir, "results.txt"), "a") as f:
        f.write(json.dumps(results) + "\n")

    return results


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python test_finetune_downstream.py`
Expected: `test_finetune_downstream.py: all checks passed`

- [ ] **Step 5: Commit**

```bash
git add finetune_downstream.py test_finetune_downstream.py
git commit -m "add finetune_downstream.py: per-dataset downstream fine-tuning driver"
```

- [ ] **Step 6: Real smoke test on the smallest dataset**

The driver's orchestration (data loading + training loop + checkpoint reload + AUC report) isn't covered by the unit tests above — they only cover the pure helpers. Verify it end-to-end for real, cheaply, before trusting it for all 12 datasets: `breastmnist` is the smallest MedMNIST-2D dataset (546 train / 78 val / 156 test images), so 1 epoch finishes in well under a minute.

Run: `python finetune_downstream.py --dataset breastmnist --epochs 1`

Expected:
- No traceback.
- Console shows `Loaded teacher backbone from ... with msg: ...` (from `load_finetune_backbone`), then one `Epoch 0: train_loss=... val_loss=...` line, then `Loaded best checkpoint from epoch 0 ...`, then `--- Overall test mAUC: <some 0-1 float> ---`.
- `Models/finetune_breastmnist/best.pth.tar` exists.
- `Outputs/finetune_breastmnist/results.json` exists and contains a `test_overall_mAUC` key with a float between 0 and 1 (with only 1 epoch of fine-tuning from a strong pretrained encoder, expect somewhere in the 0.7-0.85 range based on epoch-20 pretraining's own breastmnist reading of 0.83 teacher mAUC — it does not need to beat that, this is just a wiring smoke test, not a quality bar).

If this fails, fix the driver script directly (no test to update — this step is verifying glue code, not pure logic) and rerun until it passes.

- [ ] **Step 7: Commit the smoke-test artifacts if the run succeeded**

```bash
git add -f Models/finetune_breastmnist/best.pth.tar Outputs/finetune_breastmnist/
git status
```

Only commit if you actually want this 1-epoch smoke-test checkpoint kept in history — otherwise leave it untracked (it'll be overwritten by the real 20-epoch `breastmnist` run later) and skip this step.

---

## Self-Review Notes

- **Spec coverage:** all four spec bullets under "What's new" are covered — `train_downstream_epoch` (Task 1), head-stripping checkpoint load (Task 2), the driver's args/loading/loop/early-stop/report/output-layout (Task 3). "Out of scope" items (joint 12-dataset loop, crash-proof resume, linear probe, non-MedMNIST datasets) are none of them built — confirmed by re-reading Task 3's script top to bottom.
- **Type/signature consistency:** `train_downstream_epoch`'s signature (Task 1) matches its call site in Task 3 exactly (positional order `model, use_head_n, dataset, data_loader_train, device, criterion, optimizer, epoch, scaler=`). `load_finetune_backbone`'s signature (Task 2) matches its Task 3 call site (`model, checkpoint_path, key=`). `evaluate`/`test_classification`/`metric_AUROC` calls in Task 3 were checked against their actual current signatures in `trainer.py`/`utils.py`, not guessed.
- **No placeholders:** every step has complete, runnable code — no TBD/TODO, no "add appropriate handling".
