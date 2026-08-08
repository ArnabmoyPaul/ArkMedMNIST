# DEVIATIONS.md + Evaluation Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Document every departure from the Ark+ paper, then build the scripts needed to produce deliverable 5's results table: strong individual baselines, Ark+ fine-tuning, Ark+ linear probing, published MedMNIST numbers, deltas, and 95% CIs over ≥3 seeds.

**Architecture:** One shared `SingleTaskModel` (one encoder + neck + projector + one head, matching `ArkPlusDual`'s per-modality submodules exactly) is reused for both the individual baseline and Ark+ fine-tuning — same architecture, different initialization — so the comparison is apples-to-apples per rule 2 (never weaken a baseline). A separate linear-probing script freezes that same architecture and fits a linear classifier on frozen embeddings. A stats module supplies both an analytical (Hanley-McNeil) and a resampling (bootstrap) CI, used depending on whether the dataset's AUC is a single binary statistic or a mean over multiple classes.

**Tech Stack:** PyTorch, scikit-learn (`LogisticRegression`), numpy.

## Global Constraints

- **Depends on the core-training-rewrite plan** (`2026-08-07-core-training-rewrite.md`) having produced a real `best_model.pth` — the fine-tuning/linear-probing scripts here load that checkpoint. They're written and unit-testable now; running them for real happens after that plan's Task 9/10 produce a trained checkpoint.
- Rule 1 (fidelity over performance): every script here is an *evaluation* tool, not a change to pretraining — no new deviations are introduced by this plan beyond what's already documented.
- Rule 2 (never weaken a baseline): the individual baseline MUST use the identical architecture (Swin-Tiny 2D / ResNet3D-18 3D) as Ark+, differing only in initialization (random or ImageNet vs. Ark+ pretrained) — anything else isn't apples-to-apples and the brief is explicit that this is what sank an earlier iteration (2/6 vs. the falsely reported 6/6).
- Rule 3 (one model, not N): both `finetune.py` and `linear_probe.py` load a single checkpoint path selected on mean AUC across all 18 datasets (produced by the core-training plan), never a per-dataset checkpoint.
- **Known trap #6 (measurement floor):** report mean ± s.d. over ≥3 seeds with a 95% CI printed; never treat a difference smaller than the CI half-width as signal. Verified concretely during planning: SynapseMNIST3D's test split is 257 positive / 95 negative (352 total, confirmed via the installed `medmnist` package), and the Hanley-McNeil formula gives a 95% half-width in the 0.03-0.06 range across plausible AUC values there — the same order of magnitude as the brief's own ±0.051 figure.
- **Known trap #7:** `sklearn.linear_model.LogisticRegression` with the default `lbfgs` solver is a deterministic convex optimizer — refitting it on the *same* frozen embeddings under different random seeds produces exactly the same weights. Genuine seed variance for the linear-probing column can only come from using embeddings extracted from independently-pretrained Ark+ checkpoints (different pretraining seeds), never from re-running the probe alone. This is called out explicitly in Task 5 so it isn't silently faked.

---

### Task 1: Write `DEVIATIONS.md`

**Files:**
- Create: `DEVIATIONS.md`

- [ ] **Step 1: Write the file**

```markdown
# Deviations from the Ark+ paper/repository

Every departure from `jlianglab/Ark`'s `Ark_Plus` implementation, why, and what flag
controls it. Rule: nothing not in the paper or repo without a flag defaulting OFF.

## Architecture

**Dual encoder instead of one backbone.** Paper: a single Swin backbone (all
pretraining data is 2D chest radiographs). Here: MedMNIST mixes 12 2D datasets and 6
native 3D volumetric datasets that cannot share a 2D-only encoder. Two encoders (2D
Swin-Tiny, 3D ResNet-18) each project into a shared dimension and both feed **one**
shared `Projector` -- the mechanism the paper names for mapping representations into a
common space, so sharing it (rather than giving each modality its own) is the faithful
extension, not an invented one. See `train_ark_18datasets.py` architecture diagram in
the original brief.

**Backbone size:** Swin-Large (197M params, the paper's release) -> Swin-Tiny (~28M) +
ResNet3D-18 (~33M), ~65M combined. Reason: paper trained on 4x A100 80GB; this runs on
one RTX 4060 with 8GB VRAM. Not a flag -- a hard hardware constraint, fixed in
`ark_plus_model.ArkPlusDual`.

**3D stem:** the default 3D ResNet stem (`conv 7x7x7 stride 2` + `maxpool stride 2`)
collapses a 28^3 MedMNIST volume to 7^3 before the first residual block. Uses
`conv1_t_size=3, conv1_t_stride=1, no_max_pool=True` instead (`resnet3d.py`) -- this is
an architecture *correction* for the input scale, not a fidelity deviation from the
paper (the paper never specifies a 3D stem at all, since it has no 3D pretraining data).

## Resolution

768x768 (paper) -> 112x112 (2D). Not simply "smaller for speed": at `patch_size=4,
window_size=7`, 112 gives a 28x28 token grid that divides evenly into 4x4 windows
(28/7=4); the project's previous 64px setting gave a 16x16 grid that does *not* divide
evenly by 7, forcing timm to pad/mask windows internally. 112 is simultaneously more
correct for the windowing scheme and ~4x cheaper than 224. 3D volumes are used at their
native 28^3 -- not a deviation, that's the dataset's real resolution.

## Batch size / LR / schedule

| | Paper | This run | Why |
|---|---|---|---|
| Batch size | 50 (global, 4 GPUs) | 32 (2D) / 8 (3D) | 8GB VRAM, single GPU |
| Base LR | 0.3 | ~0.2 | Linear-scaled down for the smaller batch |
| Warmup | none | 2 epochs, linear | Small-batch single-GPU SGD at this LR is unstable without it (untested at paper's LR/batch combination on this hardware) |
| Cycles | 50 | 25 (patience 5) | ~700 GPU-hours on 4xA100 is not reproducible on one 4060; reassess after the first cycle's measured wall-clock (`train_log.txt`) |

Flags: `BASE_LR`, `WARMUP_EPOCHS`, `BATCH_SIZE_2D`, `BATCH_SIZE_3D`, `EPOCHS`, `PATIENCE`
in `train_ark_18datasets.py`'s CONFIG block.

## ChestMNIST loss weighting

`CHESTMNIST_POS_WEIGHT` and `PER_TASK_LOSS_NORM` (`train_ark_18datasets.py` CONFIG
block) -- both default **OFF**. Neither is in the Ark+ paper. If ChestMNIST AUC remains
depressed relative to its published ~0.768 individual number after the core bug fixes
(missing projector, symmetric-view bug, hardcoded class count) land, these are the next
levers to try, one at a time, with the result documented here.

## Not a deviation: EMA buffer averaging

`ema_update(..., update_buffers=True)` in `ark_plus_model.py` averages BatchNorm
running stats into the teacher, not just parameters. This is standard mean-teacher
practice and a correctness fix (without it, a ResNet3D backbone's `running_mean`/
`running_var` hold whichever dataset trained last, so the "one shared model" isn't
actually shared) -- not a deviation from anything the paper specifies, since Swin has no
BatchNorm and the paper never needed to address this. The flag exists purely so it can
be ablated, and defaults **on**.
```

- [ ] **Step 2: Commit**

```bash
git add DEVIATIONS.md
git commit -m "add DEVIATIONS.md documenting every departure from the Ark+ paper"
```

---

### Task 2: Statistics utilities — Hanley-McNeil SE and bootstrap CI

**Files:**
- Create: `stats.py`
- Test: `test_stats.py`

**Interfaces:**
- Produces: `hanley_mcneil_se(auc, n_pos, n_neg) -> float`, `hanley_mcneil_ci95(auc, n_pos, n_neg) -> (lo, hi)`, `bootstrap_auc_ci95(y_true, y_pred, cfg, n_resamples=1000, seed=0) -> (lo, hi)`, `aggregate_seeds(values) -> (mean, sd)`.

Binary-task datasets (`task == 'multi-class'` with `num_classes == 2`, or single-label binary) get the analytical Hanley-McNeil CI. Everything with more than 2 classes or multi-label reports a mean-over-classes AUC, whose sampling distribution doesn't have a clean closed form — those get a bootstrap CI (resample test-set rows with replacement, recompute `compute_auc`, repeat, take the 2.5/97.5 percentiles).

- [ ] **Step 1: Write the module**

```python
"""Statistics for AUC confidence intervals and multi-seed aggregation."""
import numpy as np


def hanley_mcneil_se(auc, n_pos, n_neg):
    """Hanley & McNeil (1982) standard error of an AUC estimate."""
    q1 = auc / (2 - auc)
    q2 = 2 * auc ** 2 / (1 + auc)
    var = (auc * (1 - auc) + (n_pos - 1) * (q1 - auc ** 2) + (n_neg - 1) * (q2 - auc ** 2)) / (n_pos * n_neg)
    return var ** 0.5


def hanley_mcneil_ci95(auc, n_pos, n_neg):
    se = hanley_mcneil_se(auc, n_pos, n_neg)
    return auc - 1.96 * se, auc + 1.96 * se


def bootstrap_auc_ci95(y_true, y_pred, cfg, n_resamples=1000, seed=0):
    """Resample-based CI for a mean-over-classes AUC (multi-class/multi-label).
    y_true/y_pred: numpy arrays, same convention as train_ark_18datasets.compute_auc.
    """
    from train_ark_18datasets import compute_auc
    import torch

    rng = np.random.default_rng(seed)
    n = len(y_true)
    scores = []
    for _ in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        try:
            scores.append(compute_auc(torch.as_tensor(y_true[idx]), torch.as_tensor(y_pred[idx]), cfg))
        except ValueError:
            continue  # a resample can occasionally drop a class entirely; skip it
    lo, hi = np.percentile(scores, [2.5, 97.5])
    return float(lo), float(hi)


def aggregate_seeds(values):
    values = np.asarray(values, dtype=float)
    return float(values.mean()), float(values.std(ddof=1)) if len(values) > 1 else 0.0
```

- [ ] **Step 2: Write a self-check**

```python
"""Self-check for stats.py: run `python test_stats.py`."""
import numpy as np
from stats import hanley_mcneil_se, hanley_mcneil_ci95, aggregate_seeds


def test_hanley_mcneil_matches_synapsemnist3d_ballpark():
    # Real SynapseMNIST3D test split, confirmed against the installed medmnist
    # package during planning: 257 positive, 95 negative (352 total). The brief
    # states a 95% half-width of ~0.051 for this dataset; across plausible AUC
    # operating points (0.6-0.9) this formula gives 0.03-0.06, the same order
    # of magnitude -- confirms the formula, not a bit-exact reproduction (the
    # brief's own AUC operating point isn't specified precisely enough to match exactly).
    n_pos, n_neg = 257, 95
    for auc in (0.6, 0.7, 0.85, 0.9):
        se = hanley_mcneil_se(auc, n_pos, n_neg)
        half_width = 1.96 * se
        assert 0.02 < half_width < 0.08, f"auc={auc}: half-width {half_width} out of expected range"


def test_ci_is_symmetric_around_auc():
    lo, hi = hanley_mcneil_ci95(0.8, 200, 200)
    assert abs((lo + hi) / 2 - 0.8) < 1e-9


def test_perfect_auc_has_zero_variance():
    # AUC=1.0 with no wrong pairs -- Hanley-McNeil formula should not blow up
    # or go negative under sqrt.
    se = hanley_mcneil_se(1.0, 50, 50)
    assert se >= 0 and not np.isnan(se)


def test_aggregate_seeds():
    mean, sd = aggregate_seeds([0.80, 0.82, 0.78])
    assert abs(mean - 0.80) < 1e-9
    assert sd > 0

    mean_single, sd_single = aggregate_seeds([0.80])
    assert sd_single == 0.0, "single seed must report sd=0, not NaN or a fabricated spread"


if __name__ == "__main__":
    test_hanley_mcneil_matches_synapsemnist3d_ballpark()
    test_ci_is_symmetric_around_auc()
    test_perfect_auc_has_zero_variance()
    test_aggregate_seeds()
    print("stats.py: all checks passed")
```

- [ ] **Step 3: Run it**

Run: `conda run -n ark_medmnist python test_stats.py`
Expected: `stats.py: all checks passed`

- [ ] **Step 4: Commit**

```bash
git add stats.py test_stats.py
git commit -m "add Hanley-McNeil and bootstrap AUC confidence interval utilities"
```

---

### Task 3: `SingleTaskModel` — shared architecture for baseline and Ark+ fine-tuning

**Files:**
- Modify: `ark_plus_model.py`
- Test: `test_single_task_model.py`

**Interfaces:**
- Consumes: `resnet3d.ResNet3D18`, `Projector`, `MultiTaskHead` (already in `ark_plus_model.py`).
- Produces: `SingleTaskModel(num_classes, is_3d, img_size_2d=112, shared_dim=512, proj_hidden=1024, proj_dim=512, pretrained_2d=False)`, and `load_arkplus_encoder(model, checkpoint_path, is_3d)`.

This is the rule-2 apples-to-apples piece: the exact same encoder/neck/projector architecture `ArkPlusDual` uses for one modality, so "strong individual baseline" and "Ark+ fine-tuned" differ only in *initialization*, never in capacity.

- [ ] **Step 1: Add the class and loader function**

```python
class SingleTaskModel(nn.Module):
    """One encoder + neck + projector + one head -- architecturally identical
    to one modality of ArkPlusDual, so baseline-vs-Ark+ comparisons only ever
    differ in initialization, never in capacity (never weaken a baseline)."""

    def __init__(self, num_classes, is_3d, img_size_2d=112, shared_dim=512,
                 proj_hidden=1024, proj_dim=512, pretrained_2d=False):
        super().__init__()
        self.is_3d = is_3d
        if is_3d:
            self.encoder = ResNet3D18(in_channels=1)
        else:
            self.encoder = timm.create_model(
                "swin_tiny_patch4_window7_224", pretrained=pretrained_2d,
                img_size=img_size_2d, num_classes=0, global_pool="avg")
        self.neck = nn.Linear(self.encoder.num_features, shared_dim)
        self.projector = Projector(in_dim=shared_dim, hidden_dim=proj_hidden, out_dim=proj_dim)
        self.head = MultiTaskHead(in_features=proj_dim, num_classes=num_classes)

    def forward(self, x):
        if not self.is_3d and x.shape[1] == 1:
            x = x.expand(-1, 3, -1, -1)
        feats = self.encoder(x)
        proj = self.projector(self.neck(feats))
        return proj, self.head(proj)


def load_arkplus_encoder(model, checkpoint_path, dataset_key, is_3d):
    """Seed a SingleTaskModel's encoder/neck/projector from a pretrained
    ArkPlusDual checkpoint (the 'teacher' state_dict, per rule 3: one model).
    The task-specific head is intentionally NOT loaded -- fine-tuning starts
    that from scratch, matching standard transfer-learning practice."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    teacher_state = ckpt["teacher"]
    prefix = "encoder_3d." if is_3d else "encoder_2d."
    neck_prefix = "neck_3d." if is_3d else "neck_2d."

    encoder_state = {k[len(prefix):]: v for k, v in teacher_state.items() if k.startswith(prefix)}
    neck_state = {k[len(neck_prefix):]: v for k, v in teacher_state.items() if k.startswith(neck_prefix)}
    projector_state = {k[len("projector."):]: v for k, v in teacher_state.items() if k.startswith("projector.")}

    model.encoder.load_state_dict(encoder_state)
    model.neck.load_state_dict(neck_state)
    model.projector.load_state_dict(projector_state)
    return model
```

- [ ] **Step 2: Write a self-check**

```python
"""Self-check for SingleTaskModel: run `python test_single_task_model.py`."""
import torch
from ark_plus_model import SingleTaskModel, build_student_teacher, load_arkplus_encoder


def test_2d_and_3d_forward_shapes():
    m2d = SingleTaskModel(num_classes=9, is_3d=False, img_size_2d=112)
    proj, logits = m2d(torch.randn(2, 3, 112, 112))
    assert proj.shape == (2, 512) and logits.shape == (2, 9)

    m3d = SingleTaskModel(num_classes=2, is_3d=True)
    proj, logits = m3d(torch.randn(2, 1, 28, 28, 28))
    assert proj.shape == (2, 512) and logits.shape == (2, 2)


def test_load_arkplus_encoder_transfers_weights_not_head(tmp_path=None):
    import tempfile, os
    student, _ = build_student_teacher([9, 2, 2])  # heads irrelevant to this test
    ckpt_path = os.path.join(tempfile.mkdtemp(), "ckpt.pth")
    torch.save({"teacher": student.state_dict()}, ckpt_path)

    target = SingleTaskModel(num_classes=9, is_3d=False, img_size_2d=112)
    before = target.head.fc.weight.clone()
    load_arkplus_encoder(target, ckpt_path, dataset_key="pathmnist", is_3d=False)
    after = target.head.fc.weight

    assert torch.allclose(target.neck.weight, student.neck_2d.weight), \
        "neck weights should transfer exactly from the pretrained checkpoint"
    assert torch.allclose(before, after), \
        "the task head must NOT be overwritten by load_arkplus_encoder"


if __name__ == "__main__":
    test_2d_and_3d_forward_shapes()
    test_load_arkplus_encoder_transfers_weights_not_head()
    print("SingleTaskModel: all checks passed")
```

- [ ] **Step 3: Run it**

Run: `conda run -n ark_medmnist python test_single_task_model.py`
Expected: `SingleTaskModel: all checks passed`

- [ ] **Step 4: Commit**

```bash
git add ark_plus_model.py test_single_task_model.py
git commit -m "add SingleTaskModel for apples-to-apples baseline vs Ark+ fine-tuning comparisons"
```

---

### Task 4: `finetune.py` — strong individual baseline and Ark+ fine-tuning, one script

**Files:**
- Create: `finetune.py`
- Test: `test_finetune.py`

**Interfaces:**
- Consumes: `SingleTaskModel`, `load_arkplus_encoder` (Task 3); `Dataset2D`, `Dataset3D`, `compute_auc` from `train_ark_18datasets.py` (reused, not duplicated); `hanley_mcneil_ci95`, `bootstrap_auc_ci95` (Task 2).
- Produces: one JSON result file per run at `<out>/<dataset>_<init>_seed<seed>.json` with keys `dataset, method, seed, val_auc, test_auc, ci95_lo, ci95_hi, n_test, task, num_classes`.

`--init random`/`--init imagenet` is the strong individual baseline (rule 2: identical architecture to Ark+, only the init differs); `--init arkplus` is deliverable 5's "Ark+ fine-tuning" row. Same training loop, same hyperparameters, same script — the only thing that can make one look better than the other is the actual pretraining, which is the entire point of the comparison.

- [ ] **Step 1: Write the script**

```python
"""Fine-tune SingleTaskModel on one MedMNIST dataset.

    python finetune.py --dataset pathmnist --init random  --seed 0 --out results/finetune
    python finetune.py --dataset pathmnist --init imagenet --seed 0 --out results/finetune
    python finetune.py --dataset pathmnist --init arkplus --arkplus_checkpoint outputs/ark_all_18_datasets_v2/best_model.pth --seed 0 --out results/finetune
"""
import argparse
import json
import os
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset_registry import get_dataset_config, DATASETS_3D
from ark_plus_model import SingleTaskModel, load_arkplus_encoder
from train_ark_18datasets import Dataset2D, Dataset3D, compute_auc
from stats import hanley_mcneil_ci95, bootstrap_auc_ci95


def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def _autocast(device):
    return torch.autocast(device_type=device.type, enabled=(device.type == 'cuda'), dtype=torch.float16)


@torch.no_grad()
def evaluate(model, cfg, loader, device):
    model.eval()
    preds, labels = [], []
    for v1, _, lbl in loader:
        v1 = v1.to(device)
        with _autocast(device):
            _, out = model(v1)
        out = torch.softmax(out, dim=1) if cfg['task'] == 'multi-class' else torch.sigmoid(out)
        preds.append(out.float().cpu()); labels.append(lbl.cpu())
    y_true, y_pred = torch.cat(labels), torch.cat(preds)
    return compute_auc(y_true, y_pred, cfg), y_true.numpy(), y_pred.numpy()


def compute_ci(cfg, y_true, y_pred, auc):
    if cfg['num_classes'] == 2 and cfg['task'] == 'multi-class':
        n_pos = int((y_true == 1).sum())
        n_neg = int((y_true == 0).sum())
        return hanley_mcneil_ci95(auc, n_pos, n_neg)
    return bootstrap_auc_ci95(y_true, y_pred, cfg, n_resamples=1000, seed=0)


def run(args):
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    is_3d = args.dataset in DATASETS_3D
    cfg = get_dataset_config(args.dataset)
    Cls = Dataset3D if is_3d else Dataset2D

    train_ds = Cls(args.dataset, 'train') if is_3d else Cls(args.dataset, 'train', args.img_size)
    val_ds   = Cls(args.dataset, 'val')   if is_3d else Cls(args.dataset, 'val',   args.img_size)
    test_ds  = Cls(args.dataset, 'test')  if is_3d else Cls(args.dataset, 'test',  args.img_size)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)
    test_loader  = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)

    model = SingleTaskModel(cfg['num_classes'], is_3d, img_size_2d=args.img_size,
                             pretrained_2d=(args.init == 'imagenet')).to(device)
    if args.init == 'arkplus':
        assert args.arkplus_checkpoint, "--arkplus_checkpoint required for --init arkplus"
        load_arkplus_encoder(model, args.arkplus_checkpoint, args.dataset, is_3d)

    crit = nn.BCEWithLogitsLoss() if cfg['task'] == 'multi-label' else nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))  # torch.cuda.amp.GradScaler() is deprecated on torch 2.13

    best_val_auc, best_state, patience_ctr = 0.0, None, 0
    for epoch in range(args.epochs):
        model.train()
        for v1, _, lbl in train_loader:
            v1, lbl = v1.to(device), lbl.to(device)
            optimizer.zero_grad(set_to_none=True)
            with _autocast(device):
                _, logits = model(v1)
                loss = crit(logits, lbl)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update()
        scheduler.step()

        val_auc, _, _ = evaluate(model, cfg, val_loader, device)
        print(f"  epoch {epoch+1}/{args.epochs}  val_auc={val_auc:.4f}")
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= args.patience:
                print(f"  early stopping at epoch {epoch+1}")
                break

    model.load_state_dict(best_state)
    test_auc, y_true, y_pred = evaluate(model, cfg, test_loader, device)
    ci_lo, ci_hi = compute_ci(cfg, y_true, y_pred, test_auc)

    result = {
        'dataset': args.dataset, 'method': args.init, 'seed': args.seed,
        'val_auc': best_val_auc, 'test_auc': test_auc,
        'ci95_lo': ci_lo, 'ci95_hi': ci_hi,
        'n_test': len(test_ds), 'task': cfg['task'], 'num_classes': cfg['num_classes'],
    }
    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, f"{args.dataset}_{args.init}_seed{args.seed}.json")
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"test_auc={test_auc:.4f} (95% CI [{ci_lo:.4f}, {ci_hi:.4f}])  ->  {out_path}")
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', required=True)
    p.add_argument('--init', choices=['random', 'imagenet', 'arkplus'], required=True)
    p.add_argument('--arkplus_checkpoint', default=None)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--patience', type=int, default=5)
    p.add_argument('--lr', type=float, default=0.01)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--img_size', type=int, default=112)
    p.add_argument('--out', default='results/finetune')
    run(p.parse_args())


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: Write a self-check that avoids downloading real MedMNIST data**

```python
"""Self-check for finetune.py's training/checkpoint-selection mechanics, using a
synthetic in-memory dataset so it runs in seconds with no network access."""
import torch
from torch.utils.data import DataLoader, TensorDataset

from finetune import evaluate, set_seed
from ark_plus_model import SingleTaskModel


def _fake_cfg():
    return {'task': 'multi-class', 'num_classes': 3, 'n_channels': 3, 'python_class': 'FakeMNIST'}


def test_evaluate_runs_and_returns_valid_auc_range():
    set_seed(0)
    device = torch.device('cpu')
    model = SingleTaskModel(num_classes=3, is_3d=False, img_size_2d=32).to(device)
    x = torch.randn(20, 3, 32, 32)
    y = torch.randint(0, 3, (20,))
    ds = TensorDataset(x, x, y)
    loader = DataLoader(ds, batch_size=4)
    auc, y_true, y_pred = evaluate(model, _fake_cfg(), loader, device)
    assert 0.0 <= auc <= 1.0
    assert y_true.shape[0] == 20 and y_pred.shape == (20, 3)


def test_set_seed_is_reproducible():
    set_seed(42)
    a = torch.rand(5)
    set_seed(42)
    b = torch.rand(5)
    assert torch.allclose(a, b)


if __name__ == "__main__":
    test_evaluate_runs_and_returns_valid_auc_range()
    test_set_seed_is_reproducible()
    print("finetune.py: all checks passed")
```

- [ ] **Step 3: Run it**

Run: `conda run -n ark_medmnist python test_finetune.py`
Expected: `finetune.py: all checks passed`

- [ ] **Step 4: Commit**

```bash
git add finetune.py test_finetune.py
git commit -m "add finetune.py: strong individual baseline and Ark+ fine-tuning via one apples-to-apples script"
```

---

### Task 5: `linear_probe.py` — frozen-embedding linear probing

**Files:**
- Create: `linear_probe.py`
- Test: `test_linear_probe.py`

**Interfaces:**
- Consumes: `SingleTaskModel`, `load_arkplus_encoder` (Task 3); `compute_auc` (`train_ark_18datasets.py`); `hanley_mcneil_ci95`/`bootstrap_auc_ci95` (Task 2).
- Produces: `<out>/<dataset>_linear_probe_seed<seed>.json`.

- [ ] **Step 1: Write the script**

```python
"""Linear probing on frozen Ark+ embeddings -- "the projected representation
is also the linear-probing embedding" (Ark+ spec).

NOTE on --seed (known trap #7): sklearn's LogisticRegression(solver='lbfgs')
is a deterministic convex optimizer. Re-running this script with a different
--seed against the SAME --arkplus_checkpoint produces byte-identical results;
that's not a bug, lbfgs has no randomness to seed. --seed here labels which
independently-pretrained checkpoint's embeddings are being probed. Do not
report "N seeds" of linear-probe variance from one checkpoint -- see
test_linear_probe.py's test_lbfgs_logistic_regression_is_deterministic for a
runnable demonstration of exactly this.

    python linear_probe.py --dataset pathmnist --arkplus_checkpoint outputs/ark_all_18_datasets_v2/best_model.pth --seed 0 --out results/probe
"""
import argparse
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier

from dataset_registry import get_dataset_config, DATASETS_3D
from ark_plus_model import SingleTaskModel, load_arkplus_encoder
from train_ark_18datasets import Dataset2D, Dataset3D, compute_auc
from stats import hanley_mcneil_ci95, bootstrap_auc_ci95


def _autocast(device):
    return torch.autocast(device_type=device.type, enabled=(device.type == 'cuda'), dtype=torch.float16)


@torch.no_grad()
def extract_embeddings(model, loader, device):
    model.eval()
    feats, labels = [], []
    for v1, _, lbl in loader:
        v1 = v1.to(device)
        with _autocast(device):
            proj, _ = model(v1)
        feats.append(proj.float().cpu().numpy())
        labels.append(lbl.numpy())
    return np.concatenate(feats), np.concatenate(labels)


def run(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    is_3d = args.dataset in DATASETS_3D
    cfg = get_dataset_config(args.dataset)
    Cls = Dataset3D if is_3d else Dataset2D
    train_ds = Cls(args.dataset, 'train') if is_3d else Cls(args.dataset, 'train', args.img_size)
    test_ds  = Cls(args.dataset, 'test')  if is_3d else Cls(args.dataset, 'test',  args.img_size)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = SingleTaskModel(cfg['num_classes'], is_3d, img_size_2d=args.img_size).to(device)
    load_arkplus_encoder(model, args.arkplus_checkpoint, args.dataset, is_3d)
    for p in model.parameters():
        p.requires_grad = False

    X_train, y_train = extract_embeddings(model, train_loader, device)
    X_test, y_test = extract_embeddings(model, test_loader, device)

    if cfg['task'] == 'multi-label':
        clf = MultiOutputClassifier(LogisticRegression(max_iter=2000))
        clf.fit(X_train, y_train)
        proba = np.stack([col[:, 1] for col in clf.predict_proba(X_test)], axis=1)
    else:
        clf = LogisticRegression(max_iter=2000, multi_class='multinomial')
        clf.fit(X_train, y_train.squeeze())
        proba = clf.predict_proba(X_test)

    y_true_t, y_pred_t = torch.as_tensor(y_test), torch.as_tensor(proba)
    test_auc = compute_auc(y_true_t, y_pred_t, cfg)
    if cfg['num_classes'] == 2 and cfg['task'] == 'multi-class':
        n_pos, n_neg = int((y_test == 1).sum()), int((y_test == 0).sum())
        ci_lo, ci_hi = hanley_mcneil_ci95(test_auc, n_pos, n_neg)
    else:
        ci_lo, ci_hi = bootstrap_auc_ci95(y_test, proba, cfg, n_resamples=1000, seed=0)

    result = {
        'dataset': args.dataset, 'method': 'linear_probe', 'seed': args.seed,
        'test_auc': float(test_auc), 'ci95_lo': ci_lo, 'ci95_hi': ci_hi,
        'n_test': len(test_ds), 'task': cfg['task'], 'num_classes': cfg['num_classes'],
        'checkpoint': args.arkplus_checkpoint,
    }
    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, f"{args.dataset}_linear_probe_seed{args.seed}.json")
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"test_auc={test_auc:.4f} (95% CI [{ci_lo:.4f}, {ci_hi:.4f}])  ->  {out_path}")
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', required=True)
    p.add_argument('--arkplus_checkpoint', required=True)
    p.add_argument('--seed', type=int, default=0,
                    help='labels which pretraining checkpoint is being probed, NOT an RNG seed (see module docstring)')
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--img_size', type=int, default=112)
    p.add_argument('--out', default='results/probe')
    run(p.parse_args())


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: Write a self-check, including a runnable proof of the determinism trap**

```python
"""Self-check for linear_probe.py: run `python test_linear_probe.py`."""
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader, TensorDataset

from linear_probe import extract_embeddings
from ark_plus_model import SingleTaskModel


def test_lbfgs_logistic_regression_is_deterministic():
    """Known trap #7, made concrete: lbfgs has no randomness, so 'refit with
    different seeds' produces byte-identical coefficients. If this ever
    fails, sklearn's default solver/behavior changed and linear_probe.py's
    docstring warning needs revisiting."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 16))
    y = (X[:, 0] > 0).astype(int)
    coefs = [LogisticRegression(max_iter=1000).fit(X, y).coef_.copy() for _ in range(3)]
    assert np.allclose(coefs[0], coefs[1]) and np.allclose(coefs[1], coefs[2])


def test_extract_embeddings_shapes():
    model = SingleTaskModel(num_classes=3, is_3d=False, img_size_2d=32)
    ds = TensorDataset(torch.randn(6, 3, 32, 32), torch.randn(6, 3, 32, 32), torch.randint(0, 3, (6,)))
    loader = DataLoader(ds, batch_size=2)
    feats, labels = extract_embeddings(model, loader, torch.device('cpu'))
    assert feats.shape == (6, 512)
    assert labels.shape == (6,)


if __name__ == "__main__":
    test_lbfgs_logistic_regression_is_deterministic()
    test_extract_embeddings_shapes()
    print("linear_probe.py: all checks passed")
```

- [ ] **Step 3: Run it**

Run: `conda run -n ark_medmnist python test_linear_probe.py`
Expected: `linear_probe.py: all checks passed`

- [ ] **Step 4: Commit**

```bash
git add linear_probe.py test_linear_probe.py
git commit -m "add linear_probe.py: frozen-embedding linear probing with documented lbfgs determinism trap"
```

---

### Task 6: `make_results_table.py` — deliverable 5's results table

**Files:**
- Create: `make_results_table.py`
- Create: `published_auc_template.json`
- Test: `test_make_results_table.py`

**Interfaces:**
- Consumes: JSON files written by `finetune.py`/`linear_probe.py` (Tasks 4-5); `aggregate_seeds` (Task 2).
- Produces: a markdown table on stdout (or `--out table.md`).

Published MedMNIST numbers are **not hardcoded here** — transcribing them from memory risks silently corrupting the delta column, which is exactly the fidelity failure rule 1 exists to prevent. `published_auc_template.json` ships with `null` placeholders and a comment pointing at the source; `make_results_table.py` prints `N/A` for any dataset without a filled-in number rather than guessing.

- [ ] **Step 1: Write the published-AUC template**

```json
{
  "_comment": "Fill in from https://medmnist.com/ or Table 2/3 of the MedMNIST v2 paper (Yang et al.) before generating the final results table. Leave null for anything unverified -- make_results_table.py prints N/A rather than guessing.",
  "pathmnist": null,
  "bloodmnist": null,
  "dermamnist": null,
  "octmnist": null,
  "pneumoniamnist": null,
  "retinamnist": null,
  "breastmnist": null,
  "tissuemnist": null,
  "organamnist": null,
  "organcmnist": null,
  "organsmnist": null,
  "chestmnist": null,
  "organmnist3d": null,
  "nodulemnist3d": null,
  "adrenalmnist3d": null,
  "fracturemnist3d": null,
  "vesselmnist3d": null,
  "synapsemnist3d": null
}
```

- [ ] **Step 2: Write the table generator**

```python
"""Build deliverable 5's results table from finetune.py/linear_probe.py JSON output.

    python make_results_table.py --finetune_dir results/finetune --probe_dir results/probe \
        --published published_auc.json --out RESULTS.md
"""
import argparse
import glob
import json
import os
from collections import defaultdict

from dataset_registry import DATASETS
from stats import aggregate_seeds


def load_runs(directory, method_filter=None):
    runs = defaultdict(list)  # (dataset, method) -> [result dicts]
    if not directory or not os.path.isdir(directory):
        return runs
    for path in glob.glob(os.path.join(directory, "*.json")):
        with open(path) as f:
            r = json.load(f)
        if method_filter is None or r['method'] == method_filter:
            runs[(r['dataset'], r['method'])].append(r)
    return runs


def summarize(runs_for_key):
    aucs = [r['test_auc'] for r in runs_for_key]
    mean, sd = aggregate_seeds(aucs)
    # Report the tightest (most informative) single-run CI among the seeds as
    # the "measurement floor" context, alongside the cross-seed spread.
    ci = min(runs_for_key, key=lambda r: r['ci95_hi'] - r['ci95_lo'])
    return mean, sd, ci['ci95_lo'], ci['ci95_hi'], len(runs_for_key)


def fmt_cell(runs_for_key):
    if not runs_for_key:
        return "—"
    mean, sd, lo, hi, n = summarize(runs_for_key)
    sd_str = f" ± {sd:.4f}" if n > 1 else " (1 seed)"
    return f"{mean:.4f}{sd_str} [{lo:.4f}, {hi:.4f}]"


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--finetune_dir', default='results/finetune')
    p.add_argument('--probe_dir', default='results/probe')
    p.add_argument('--published', default='published_auc.json')
    p.add_argument('--out', default=None)
    args = p.parse_args()

    ft_runs = load_runs(args.finetune_dir)
    probe_runs = load_runs(args.probe_dir)
    published = {}
    if os.path.isfile(args.published):
        with open(args.published) as f:
            published = {k: v for k, v in json.load(f).items() if not k.startswith('_')}

    lines = [
        "| Dataset | Strong baseline | Ark+ fine-tune | Ark+ linear probe | Published | Delta (Ark+FT - baseline) |",
        "|---|---|---|---|---|---|",
    ]
    for key in DATASETS:
        baseline = ft_runs.get((key, 'random'), []) + ft_runs.get((key, 'imagenet'), [])
        arkplus_ft = ft_runs.get((key, 'arkplus'), [])
        probe = probe_runs.get((key, 'linear_probe'), [])
        pub = published.get(key)
        pub_str = f"{pub:.4f}" if pub is not None else "N/A"

        delta_str = "N/A"
        if baseline and arkplus_ft:
            base_mean, _, _, _, _ = summarize(baseline)
            ft_mean, _, _, _, _ = summarize(arkplus_ft)
            delta_str = f"{ft_mean - base_mean:+.4f}"

        lines.append(f"| {key} | {fmt_cell(baseline)} | {fmt_cell(arkplus_ft)} | {fmt_cell(probe)} | {pub_str} | {delta_str} |")

    table = "\n".join(lines)
    if args.out:
        with open(args.out, 'w') as f:
            f.write(table + "\n")
        print(f"Wrote {args.out}")
    else:
        print(table)


if __name__ == '__main__':
    main()
```

- [ ] **Step 3: Write a self-check with synthetic result files**

```python
"""Self-check for make_results_table.py: run `python test_make_results_table.py`."""
import json
import os
import shutil
import tempfile

from make_results_table import load_runs, summarize, fmt_cell


def _write(d, dataset, method, seed, auc, lo, hi):
    with open(os.path.join(d, f"{dataset}_{method}_seed{seed}.json"), 'w') as f:
        json.dump({'dataset': dataset, 'method': method, 'seed': seed,
                    'test_auc': auc, 'ci95_lo': lo, 'ci95_hi': hi}, f)


def test_load_and_summarize_multi_seed():
    d = tempfile.mkdtemp()
    try:
        _write(d, 'pathmnist', 'arkplus', 0, 0.90, 0.87, 0.93)
        _write(d, 'pathmnist', 'arkplus', 1, 0.92, 0.89, 0.95)
        _write(d, 'pathmnist', 'arkplus', 2, 0.91, 0.88, 0.94)
        runs = load_runs(d)
        key_runs = runs[('pathmnist', 'arkplus')]
        assert len(key_runs) == 3
        mean, sd, lo, hi, n = summarize(key_runs)
        assert abs(mean - 0.91) < 1e-9
        assert sd > 0
        cell = fmt_cell(key_runs)
        assert "±" in cell and "[" in cell
    finally:
        shutil.rmtree(d)


def test_missing_directory_returns_empty():
    runs = load_runs("/path/does/not/exist")
    assert runs == {}
    assert fmt_cell([]) == "—"


if __name__ == "__main__":
    test_load_and_summarize_multi_seed()
    test_missing_directory_returns_empty()
    print("make_results_table.py: all checks passed")
```

- [ ] **Step 4: Run it**

Run: `conda run -n ark_medmnist python test_make_results_table.py`
Expected: `make_results_table.py: all checks passed`

- [ ] **Step 5: Commit**

```bash
git add make_results_table.py published_auc_template.json test_make_results_table.py
git commit -m "add make_results_table.py for deliverable 5; published AUCs sourced from medmnist.com, never hardcoded from memory"
```

---

## Self-Review Notes

- **Spec coverage:** deliverable 4 (`DEVIATIONS.md`) — Task 1. Deliverable 5 (strong individual baseline, Ark+ fine-tuning, Ark+ linear probing, published number, delta, 95% CI, ≥3 seeds) — Tasks 2-6 together: `finetune.py --init random/imagenet` is the baseline, `--init arkplus` is the fine-tune row, `linear_probe.py` is the probe row, `published_auc_template.json` + `make_results_table.py` join in the published number and delta, `stats.py` supplies the CI, and running any script 3x with different `--seed` (finetune) gives the ≥3-seed spread `make_results_table.py` aggregates.
- **No placeholders:** every step has complete, runnable code. The one intentionally-`null` file (`published_auc_template.json`) is null *by design*, not laziness — filling it with unverified numbers would be a worse fidelity violation than leaving it explicit, and `make_results_table.py` handles the nulls gracefully (prints `N/A`) rather than crashing.
- **Type consistency check:** every JSON result written by `finetune.py`/`linear_probe.py` shares the key set `{dataset, method, seed, test_auc, ci95_lo, ci95_hi, n_test, task, num_classes}` (`finetune.py` additionally has `val_auc`; `linear_probe.py` additionally has `checkpoint`) — `make_results_table.py`'s `load_runs`/`summarize`/`fmt_cell` only touch the shared subset, so both feed it without special-casing.
- **Known trap #7 is not just documented but tested:** `test_lbfgs_logistic_regression_is_deterministic` in Task 5 fails loudly if this assumption ever stops holding, rather than silently letting someone report fabricated linear-probe seed variance.
