# 2D Downstream Fine-Tuning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fine-tune and evaluate the completed 12-dataset 2D omni-pretrained encoder separately on each of the 12 MedMNIST-2D datasets, then report a combined 18-dataset (12 2D + 6 3D) benchmark table against the professor-supplied targets.

**Architecture:** One new driver script, `finetune_downstream.py`, structurally identical to the existing `finetune_downstream_3d.py` (same helpers, same output layout, same PASS/WITHIN_1PCT/FAIL reporting), swapped from 3D to 2D data/model plumbing (`dict_dataloarder`/`build_transform_classification` instead of `MedMNIST3DWrapper`, `build_omni_model` instead of `build_omni_model_3d`, `key="teacher"` instead of `"teacher_3d"`). No changes to any existing file — `trainer.train_downstream_epoch`, `models.load_finetune_backbone`, `models.build_omni_model` are all already built and reused as-is.

**Tech Stack:** PyTorch 2.13.0+cu130, timm 1.0.28, medmnist 3.0.2, argparse (not `main_ark.py`'s optparse — same rationale as `finetune_downstream_3d.py`: only a narrow slice of fields matters for a single-dataset fine-tune driver).

## Global Constraints

- Conda env `ark_medmnist`; run every command via `/d/anaconda/envs/ark_medmnist/python.exe` (confirmed working path, git-bash style).
- Single GPU (RTX 4060, 8188 MiB) — never run two training processes concurrently; check `nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv` is near-idle before starting a run.
- No crash-proof resume for fine-tuning (per the approved spec, `docs/superpowers/specs/2026-08-18-medmnist2d-downstream-finetune-design.md`): a single-dataset fine-tune is cheap enough to rerun from scratch if killed.
- `crop_size=112`, `resize=128`, `normalize="imagenet"` — must match the values the 2D pretrain run used (`pretrain_medmnist2d_run.py`), since the checkpoint's Swin backbone was built with `img_size=112`.
- Checkpoint default: the completed 2D pretrain run's **final epoch (24)**, no "best epoch" selection — matches the precedent already established for the 3D work and `DEVIATIONS.md`'s note that best-checkpoint selection is out of scope.

---

### Task 1: `finetune_downstream.py` — 2D downstream fine-tune driver

**Files:**
- Create: `finetune_downstream.py`
- Create: `test_finetune_downstream.py`

**Interfaces:**
- Consumes: `trainer.train_downstream_epoch(model, use_head_n, dataset, data_loader_train, device, criterion, optimizer, epoch, scaler=None) -> float`; `trainer.evaluate(model, use_head_n, data_loader_val, device, criterion, dataset, scaler=None) -> float`; `trainer.test_classification(model, use_head_n, data_loader_test, device, multiclass=False, is_3d=False) -> (y_test, p_test)`; `models.build_omni_model(args, num_classes_list) -> model`; `models.load_finetune_backbone(model, checkpoint_path, key='teacher')`; `models.save_checkpoint(state, filename='model')`; `utils.get_config(path) -> dict`; `utils.metric_AUROC(target, output, nb_classes) -> list[float]`; `dataloader.dict_dataloarder` (dict, mutated in place); `dataloader.build_transform_classification(normalize, crop_size, resize, mode, test_augment=True)`; `medmnist_dataloader.MEDMNIST_2D_KEYS` (list of 12 strings); `medmnist_dataloader.MEDMNIST_DATALOADER_DICT` (dict).
- Produces: `finetune_downstream.py`'s `BENCHMARKS_2D` (dict, 12 keys), `_compare_to_benchmark(auc, target) -> str`, `_early_stop(val_losses, patience) -> bool`, `parse_args(argv=None) -> Namespace`, `main(argv=None) -> dict` (the `results` dict) — all consumed by Task 2's CLI invocations and by `test_finetune_downstream.py`.

- [ ] **Step 1: Write the failing test file**

Create `test_finetune_downstream.py`:

```python
"""test_finetune_downstream.py — Run: python test_finetune_downstream.py"""
from finetune_downstream import _early_stop, _compare_to_benchmark, parse_args, BENCHMARKS_2D


def test_no_stop_while_still_improving():
    assert not _early_stop([0.9, 0.8, 0.7, 0.6, 0.5], patience=2)


def test_no_stop_before_patience_window_elapses():
    assert not _early_stop([0.5, 0.6, 0.7], patience=3)


def test_stops_after_patience_epochs_without_improvement():
    assert _early_stop([0.9, 0.4, 0.5, 0.6, 0.7], patience=3)


def test_stops_exactly_at_patience_boundary():
    assert _early_stop([0.5, 0.6, 0.7], patience=2)


def test_compare_pass_when_auc_meets_or_beats_target():
    assert _compare_to_benchmark(0.989, 0.989) == "PASS"
    assert _compare_to_benchmark(0.999, 0.989) == "PASS"


def test_compare_within_1pct_when_short_by_at_most_one_point():
    assert _compare_to_benchmark(0.979, 0.989) == "WITHIN_1PCT"


def test_compare_fail_when_short_by_more_than_one_point():
    assert _compare_to_benchmark(0.90, 0.989) == "FAIL"


def test_compare_no_target_generic():
    assert _compare_to_benchmark(0.80, None) == "NO_TARGET"


def test_chestmnist_target_is_0_773():
    assert BENCHMARKS_2D["chestmnist"] == 0.773


def test_all_12_keys_present():
    from medmnist_dataloader import MEDMNIST_2D_KEYS
    assert set(BENCHMARKS_2D) == set(MEDMNIST_2D_KEYS)


def test_parse_args_requires_dataset():
    try:
        parse_args([])
        assert False, "expected SystemExit for missing required --dataset"
    except SystemExit:
        pass


def test_parse_args_defaults():
    args = parse_args(["--dataset", "chestmnist"])
    assert args.dataset == "chestmnist"
    assert args.epochs == 20
    assert args.lr == 1e-3
    assert args.patience == 5
    assert args.batch_size == 32


if __name__ == "__main__":
    test_no_stop_while_still_improving()
    test_no_stop_before_patience_window_elapses()
    test_stops_after_patience_epochs_without_improvement()
    test_stops_exactly_at_patience_boundary()
    test_compare_pass_when_auc_meets_or_beats_target()
    test_compare_within_1pct_when_short_by_at_most_one_point()
    test_compare_fail_when_short_by_more_than_one_point()
    test_compare_no_target_generic()
    test_chestmnist_target_is_0_773()
    test_all_12_keys_present()
    test_parse_args_requires_dataset()
    test_parse_args_defaults()
    print("test_finetune_downstream.py: all checks passed")
```

- [ ] **Step 2: Run it to verify it fails**

Run: `/d/anaconda/envs/ark_medmnist/python.exe test_finetune_downstream.py`
Expected: `ModuleNotFoundError: No module named 'finetune_downstream'`

- [ ] **Step 3: Write `finetune_downstream.py`**

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

dict_dataloarder.update(MEDMNIST_DATALOADER_DICT)  # register the 12 MedMNIST 2D classes

DEFAULT_CHECKPOINT = (
    "Models/swin_tiny_medmnist2d_swintiny/"
    "Ark_Plus_pathmnist_bloodmnist_dermamnist_octmnist_pneumoniamnist_retinamnist_"
    "breastmnist_tissuemnist_organamnist_organcmnist_organsmnist_chestmnist/"
    "medmnist2d_swintiny/"
    "Ark_Plus_pathmnist_bloodmnist_dermamnist_octmnist_pneumoniamnist_retinamnist_"
    "breastmnist_tissuemnist_organamnist_organcmnist_organsmnist_chestmnist24.pth.tar"
)

BENCHMARKS_2D = {
    "chestmnist": 0.773,
    "pathmnist": 0.989,
    "dermamnist": 0.912,
    "octmnist": 0.958,
    "pneumoniamnist": 0.962,
    "retinamnist": 0.716,
    "breastmnist": 0.866,
    "bloodmnist": 0.997,
    "tissuemnist": 0.932,
    "organamnist": 0.998,
    "organcmnist": 0.993,
    "organsmnist": 0.975,
}


def _compare_to_benchmark(auc, target):
    """PASS if auc meets/beats target, WITHIN_1PCT if short by <=0.01,
    FAIL otherwise. NO_TARGET when the benchmark table has no entry."""
    if target is None:
        return "NO_TARGET"
    if auc >= target:
        return "PASS"
    if target - auc <= 0.01 + 1e-9:  # epsilon for float imprecision
        return "WITHIN_1PCT"
    return "FAIL"


def _early_stop(val_losses, patience):
    """True once val_losses hasn't set a new best in the last `patience` epochs."""
    if len(val_losses) <= patience:
        return False
    best_idx = min(range(len(val_losses)), key=lambda i: val_losses[i])
    return best_idx <= len(val_losses) - 1 - patience


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Downstream fine-tune a 2D MedMNIST omni-pretrained checkpoint")
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
    multiclass = ds_cfg["task_type"] == "multi-class classification"
    criterion = torch.nn.CrossEntropyLoss() if multiclass else torch.nn.BCEWithLogitsLoss()

    crop_size, resize = 112, 128
    train_set = dict_dataloarder[args.dataset](
        images_path=ds_cfg["data_dir"], file_path=ds_cfg["train_list"],
        crop_size=crop_size, resize=resize, augment=None)
    val_set = dict_dataloarder[args.dataset](
        images_path=ds_cfg["data_dir"], file_path=ds_cfg["val_list"],
        crop_size=crop_size, resize=resize,
        augment=build_transform_classification(normalize="imagenet", crop_size=crop_size,
                                                 resize=resize, mode="valid"))
    test_set = dict_dataloarder[args.dataset](
        images_path=ds_cfg["data_dir"], file_path=ds_cfg["test_list"],
        crop_size=crop_size, resize=resize,
        augment=build_transform_classification(normalize="imagenet", crop_size=crop_size,
                                                 resize=resize, mode="test", test_augment=True))

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=max(1, args.batch_size // 2), shuffle=False,
                              num_workers=args.workers, pin_memory=True)

    model_args = SimpleNamespace(model_name="swin_tiny", crop_size=crop_size,
                                  projector_features=512, use_mlp=False, pretrained_weights=None)
    model = build_omni_model(model_args, num_classes_list=[len(diseases)])
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

    best_ckpt = torch.load(best_path + ".pth.tar", map_location=device, weights_only=False)  # self-generated checkpoint, trusted
    model.load_state_dict(best_ckpt["state_dict"])
    print(f"Loaded best checkpoint from epoch {best_ckpt['epoch']} (val_loss={best_ckpt['val_loss']:.4f})")

    y_test, p_test = test_classification(model, 0, test_loader, device, multiclass, is_3d=False)
    individual_auc = metric_AUROC(y_test, p_test, len(diseases))
    overall_auc = float(np.mean(individual_auc))
    target = BENCHMARKS_2D[args.dataset]
    verdict = _compare_to_benchmark(overall_auc, target)

    results = {
        "dataset": args.dataset,
        "checkpoint": args.checkpoint,
        "best_epoch": best_ckpt["epoch"],
        "best_val_loss": best_val_loss,
        "test_auc_per_class": dict(zip(diseases, [float(a) for a in individual_auc])),
        "test_overall_mAUC": overall_auc,
        "benchmark_target": target,
        "verdict": verdict,
    }
    print(f"  Test AUC per class: {results['test_auc_per_class']}")
    print(f"  --- Overall test mAUC: {overall_auc:.4f} (target={target}) -> {verdict} ---")

    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    with open(os.path.join(output_dir, "results.txt"), "a") as f:
        f.write(json.dumps(results) + "\n")

    return results


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `/d/anaconda/envs/ark_medmnist/python.exe test_finetune_downstream.py`
Expected: `test_finetune_downstream.py: all checks passed`

- [ ] **Step 5: Real smoke run on the smallest 2D dataset**

Confirm the GPU is idle first:
`nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv`

Run (breastmnist: 546 train samples, smallest of the 12 — 2 epochs is enough to
catch wiring bugs without waiting for a full 20-epoch run):
`/d/anaconda/envs/ark_medmnist/python.exe finetune_downstream.py --dataset breastmnist --epochs 2`

Expected: completes without error, prints a line matching
`--- Overall test mAUC: 0.XXXX (target=0.866) -> (PASS|WITHIN_1PCT|FAIL) ---`,
and creates `Outputs/finetune_breastmnist/results.json`. The exact AUC number
doesn't matter here (2 epochs is not a real result) — only that the full
pipeline (data loading, checkpoint loading, training, eval, test, reporting)
runs end to end without a crash. This result gets overwritten by the real
20-epoch run in Task 2.

- [ ] **Step 6: Commit**

```bash
git add finetune_downstream.py test_finetune_downstream.py
git commit -m "$(cat <<'EOF'
Add finetune_downstream.py: 2D downstream fine-tuning driver

Mirrors finetune_downstream_3d.py for the 12 MedMNIST-2D datasets, closing
the gap left by the 2026-08-12 spec (approved but never implemented). Fine
tunes from the completed 2D omni-pretrain run's final checkpoint (epoch 24).

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: Run all 12 fine-tunes, report the combined 18-dataset table

**Files:** none created or modified (execution-only task).

**Interfaces:**
- Consumes: `finetune_downstream.py`'s CLI (`--dataset <key>`) from Task 1; `finetune_downstream_3d.py`'s existing `Outputs/finetune_<3d-dataset>/results.json` files (already on disk from the earlier 3D work) for the 6 3D rows of the combined table.
- Produces: 12 `Outputs/finetune_<2d-dataset>/results.json` files; one combined markdown table (reported to the user, not a new file — per the spec's explicit "no new report-generation script" scope).

- [ ] **Step 1: Confirm GPU is idle, then launch all 12 sequentially in the background**

```bash
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv
cd "/d/ark +"
/d/anaconda/envs/ark_medmnist/python.exe finetune_downstream.py --dataset pathmnist       > finetune_pathmnist.log 2>&1 && \
/d/anaconda/envs/ark_medmnist/python.exe finetune_downstream.py --dataset bloodmnist      > finetune_bloodmnist.log 2>&1 && \
/d/anaconda/envs/ark_medmnist/python.exe finetune_downstream.py --dataset dermamnist      > finetune_dermamnist.log 2>&1 && \
/d/anaconda/envs/ark_medmnist/python.exe finetune_downstream.py --dataset octmnist        > finetune_octmnist.log 2>&1 && \
/d/anaconda/envs/ark_medmnist/python.exe finetune_downstream.py --dataset pneumoniamnist  > finetune_pneumoniamnist.log 2>&1 && \
/d/anaconda/envs/ark_medmnist/python.exe finetune_downstream.py --dataset retinamnist     > finetune_retinamnist.log 2>&1 && \
/d/anaconda/envs/ark_medmnist/python.exe finetune_downstream.py --dataset breastmnist     > finetune_breastmnist.log 2>&1 && \
/d/anaconda/envs/ark_medmnist/python.exe finetune_downstream.py --dataset tissuemnist     > finetune_tissuemnist.log 2>&1 && \
/d/anaconda/envs/ark_medmnist/python.exe finetune_downstream.py --dataset organamnist     > finetune_organamnist.log 2>&1 && \
/d/anaconda/envs/ark_medmnist/python.exe finetune_downstream.py --dataset organcmnist     > finetune_organcmnist.log 2>&1 && \
/d/anaconda/envs/ark_medmnist/python.exe finetune_downstream.py --dataset organsmnist     > finetune_organsmnist.log 2>&1 && \
/d/anaconda/envs/ark_medmnist/python.exe finetune_downstream.py --dataset chestmnist      > finetune_chestmnist.log 2>&1
echo "ALL_2D_DONE exit=$?"
```

Run this as a single background job (it will take a while — `chestmnist` alone
has ~78K train images with `test_augment` TenCrop evaluation). Ordered smallest
to largest so failures surface early and cheaply; `chestmnist` last.

- [ ] **Step 2: Once the job completes, extract each result**

```bash
cd "/d/ark +"
for d in pathmnist bloodmnist dermamnist octmnist pneumoniamnist retinamnist \
         breastmnist tissuemnist organamnist organcmnist organsmnist chestmnist; do
  echo "=== $d ==="
  grep -E "Overall test mAUC" "finetune_${d}.log"
done
```

Expected: one `--- Overall test mAUC: 0.XXXX (target=0.YYY) -> VERDICT ---` line
per dataset, matching each dataset's `BENCHMARKS_2D` target from Task 1.

- [ ] **Step 3: Compile and report the combined 18-dataset table**

Read each of the 12 `Outputs/finetune_<dataset>/results.json` files plus the
existing 6 `Outputs/finetune_<3D-dataset>/results.json` files (already on disk
from the earlier 3D work — `OrganMNIST3D`, `VesselMNIST3D`, `AdrenalMNIST3D`,
`FractureMNIST3D`, `NoduleMNIST3D`, `SynapseMNIST3D`). Report one markdown
table to the user: dataset, target, achieved AUC, verdict, for all 18. No new
script — this is a one-time manual compilation, per the spec's explicit scope
boundary.

- [ ] **Step 4: If any 2D dataset misses its target by more than 1%**

Do not automatically retry. Report the shortfall to the user and ask whether
to pursue a tuning pass (analogous to the 3D work's bounded lr/epoch retry
procedure) — out of scope for this plan to do unprompted, since the spec's
"Out of scope" section explicitly defers any 2D pretraining/tuning follow-up.

## Self-Review

**Spec coverage:** Goal (fine-tune 12 2D datasets, report combined 18-dataset
table) → Task 2. New file `finetune_downstream.py` with all listed
differences from the 3D driver (dataset choices, checkpoint default, batch
size, data loading, model builder, `test_classification` is_3d flag,
`BENCHMARKS_2D`) → Task 1 Step 3. Reuse of existing
`trainer`/`models`/`utils` functions unchanged → Task 1's Interfaces block
(no modifications to those files anywhere in this plan). Testing
requirements (pure-helper unit tests mirroring
`test_finetune_downstream_3d.py`, real smoke run on the smallest dataset
before trusting all 12) → Task 1 Steps 1-5. Execution plan (sequential
background run, per-dataset independent failure handling, hand-compiled
combined table, no new report script) → Task 2. Out-of-scope items (no
joint 18-way retry, no 3D changes, no automated report generator, no 2D
pretrain re-tuning) → none of the tasks above touch any of those.

**Placeholder scan:** No TBD/TODO. Task 2 Steps 2-3 don't show literal
numeric results because they're genuinely unknown until the runs execute —
not a missing-detail placeholder, the *commands* and *expected format* are
fully specified.

**Type consistency:** `finetune_downstream.py`'s `main(argv=None) -> dict`
matches the `results` dict shape used in Task 2 Step 3 (`dataset`,
`checkpoint`, `best_epoch`, `best_val_loss`, `test_auc_per_class`,
`test_overall_mAUC`, `benchmark_target`, `verdict` — identical keys to
`finetune_downstream_3d.py`'s `results` dict, confirmed by direct read).
`_compare_to_benchmark`/`_early_stop`/`parse_args`/`BENCHMARKS_2D` names
used in Task 1 Step 1's test file match Task 1 Step 3's implementation
exactly (same names, same signatures).
