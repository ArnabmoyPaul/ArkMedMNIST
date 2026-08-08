# Core Training Script Rewrite Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the slice-averaged, projector-less, epoch-only-checkpointed `train_ark_18datasets.py` with a faithful Ark+ replication: native dual 2D/3D encoders feeding one shared projector, correct losses, AMP, and per-dataset crash-proof resume.

**Architecture:** Two encoders (Swin-Tiny for 2D, a corrected-stem 3D ResNet-18 for 3D) each project into a common `shared_dim` via a per-modality neck, then share ONE `Projector` MLP. Classification heads read the projector's *unnormalized* output; the consistency loss uses the *normalized* projector output. Student/teacher pair via EMA (params + buffers). Checkpointing happens after every dataset, not every epoch, with atomic writes and full RNG state.

**Tech Stack:** PyTorch 2.13 (cu130), timm (Swin-Tiny), medmnist, AMP (`torch.autocast` + `GradScaler`).

## Global Constraints

- **Rule 1 — fidelity over performance:** nothing outside the Ark+ paper/repo without a flag defaulting OFF, documented in `DEVIATIONS.md` (written in the follow-up plan).
- **Rule 2 — never weaken a baseline:** not directly implicated by this plan (no baseline training here), but the shared encoder/projector must not be crippled to make Ark+ look better.
- **Rule 3 — one model, not N:** a single checkpoint selected on the mean AUC across all 18 datasets, never per-dataset.
- GPU: RTX 4060, 8 GB VRAM. Size everything to fit with headroom.
- `medmnist.INFO` is authoritative for classes/task-type/channels — verified directly against the installed package (`medmnist==3.0.2`) during planning; never hardcode a class-count dict again.
- EMA momentum stays **0.9**, applied once per dataset-epoch (not per training step) — confirmed cadence-correct per the brief's known trap #2.
- Each bug fix (A–G) lands in its own commit naming the problem, per deliverable 3.

---

### Task 1: Corrected-stem 3D ResNet-18 backbone

**Files:**
- Create: `resnet3d.py`
- Test: `test_resnet3d.py`

**Interfaces:**
- Produces: `ResNet3D18(in_channels=1)` — `nn.Module` with `.num_features = 512` and `.forward_features(x)` where `x` is `(B, in_channels, D, H, W)` → returns `(B, 512)`.

This fixes **known trap #5**: the default 3D ResNet stem (`conv 7×7×7 stride 2` then `maxpool stride 2`) collapses a 28³ volume to 7³ before the first residual block. The corrected stem — `conv1_t_size=3, conv1_t_stride=1, no_max_pool=True` — is the exact configuration that reached 0.9099 AUC on SynapseMNIST3D in an earlier iteration of this project (best published result is 0.851). Spatial stride still halves H/W at the stem; depth is left alone since 28 is already small.

- [ ] **Step 1: Write the backbone**

```python
"""3D ResNet-18 with a stem corrected for small (28^3) volumes.

Default 3D ResNet stems (conv 7x7x7 stride 2 + maxpool stride 2) collapse a
28^3 input to 7^3 before the first residual block, destroying most spatial
signal. This stem uses a temporal (depth) kernel of 3 with stride 1, and
skips the maxpool, so depth is never downsampled at the stem -- only
height/width are, via the stem conv's stride.
"""
import torch
import torch.nn as nn


class BasicBlock3D(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv3d(in_planes, planes, kernel_size=3, stride=stride,
                                padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=1,
                                padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out = self.relu(out + identity)
        return out


class ResNet3D18(nn.Module):
    def __init__(self, in_channels=1, conv1_t_size=3, conv1_t_stride=1,
                 no_max_pool=True, widths=(64, 128, 256, 512)):
        super().__init__()
        self.no_max_pool = no_max_pool
        self.in_planes = widths[0]

        self.stem_conv = nn.Conv3d(
            in_channels, widths[0],
            kernel_size=(conv1_t_size, 7, 7),
            stride=(conv1_t_stride, 2, 2),
            padding=(conv1_t_size // 2, 3, 3),
            bias=False,
        )
        self.stem_bn = nn.BatchNorm3d(widths[0])
        self.stem_relu = nn.ReLU(inplace=True)
        if not no_max_pool:
            self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(widths[0], 2, stride=1)
        self.layer2 = self._make_layer(widths[1], 2, stride=2)
        self.layer3 = self._make_layer(widths[2], 2, stride=2)
        self.layer4 = self._make_layer(widths[3], 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.num_features = widths[3]

    def _make_layer(self, planes, blocks, stride):
        downsample = None
        if stride != 1 or self.in_planes != planes:
            downsample = nn.Sequential(
                nn.Conv3d(self.in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(planes),
            )
        layers = [BasicBlock3D(self.in_planes, planes, stride, downsample)]
        self.in_planes = planes
        for _ in range(1, blocks):
            layers.append(BasicBlock3D(self.in_planes, planes))
        return nn.Sequential(*layers)

    def forward_features(self, x):
        x = self.stem_relu(self.stem_bn(self.stem_conv(x)))
        if not self.no_max_pool:
            x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x).flatten(1)
        return x

    def forward(self, x):
        return self.forward_features(x)
```

- [ ] **Step 2: Write a self-check**

```python
"""Self-check for ResNet3D18: run `python test_resnet3d.py`."""
import torch
from resnet3d import ResNet3D18


def test_output_shape():
    model = ResNet3D18(in_channels=1)
    x = torch.randn(2, 1, 28, 28, 28)
    feats = model(x)
    assert feats.shape == (2, 512), f"expected (2, 512), got {tuple(feats.shape)}"


def test_depth_not_collapsed_at_stem():
    model = ResNet3D18(in_channels=1)
    x = torch.randn(1, 1, 28, 28, 28)
    with torch.no_grad():
        stem_out = model.stem_relu(model.stem_bn(model.stem_conv(x)))
    # stride=(1,2,2): depth must be unchanged, H/W must halve.
    assert stem_out.shape[2] == 28, f"depth changed at stem: {stem_out.shape}"
    assert stem_out.shape[3] == 14 and stem_out.shape[4] == 14, \
        f"H/W did not halve as expected: {stem_out.shape}"


def test_param_count_reasonable():
    model = ResNet3D18(in_channels=1)
    n = sum(p.numel() for p in model.parameters())
    assert 25_000_000 < n < 40_000_000, f"unexpected param count: {n}"


if __name__ == "__main__":
    test_output_shape()
    test_depth_not_collapsed_at_stem()
    test_param_count_reasonable()
    print("resnet3d.py: all checks passed")
```

- [ ] **Step 3: Run it**

Run: `conda run -n ark_medmnist python test_resnet3d.py`
Expected: `resnet3d.py: all checks passed`

- [ ] **Step 4: Commit**

```bash
git add resnet3d.py test_resnet3d.py
git commit -m "add corrected-stem 3D ResNet-18 backbone (fixes known 28^3 stem collapse)"
```

---

### Task 2: Dual-encoder model with shared projector (fixes bug D — missing projector)

**Files:**
- Modify: `ark_plus_model.py`
- Test: `test_ark_plus_model.py`

**Interfaces:**
- Consumes: `resnet3d.ResNet3D18` (Task 1).
- Produces: `ArkPlusDual(num_classes_list, img_size_2d=112, shared_dim=512, proj_hidden=1024, proj_dim=512, pretrained_2d=True)` — `.forward(x, head_n)` returns `(proj, logits)` where `proj` is the **unnormalized** shared-projector output (also the linear-probing embedding) and `logits = heads[head_n](proj)`. Modality is dispatched on `x.dim()`: `4` → 2D path, `5` → 3D path. Also produces `ema_update(student, teacher, momentum, update_buffers=True)`.

The existing `Projector` and `MultiTaskHead` classes in this file are correct and reused as-is. `ArkPlus3D`/`ArkPlusStudentTeacher` are deleted — they fold depth into batch and average (the same slicing bug as `train_ark_18datasets.py`'s `_encode`), and nothing else in the repo imports them (verified via grep).

- [ ] **Step 1: Fix the top-of-file imports, then replace the slice-based 3D classes with the dual-encoder model**

Replace the file's import block:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.swin_transformer import SwinTransformer
from einops import rearrange
```

with:

```python
import torch
import torch.nn as nn
import timm
from resnet3d import ResNet3D18
```

`from einops import rearrange` was only used by the `ArkPlus3D` class deleted in this step, and **`einops` is not installed in the `ark_medmnist` conda env** (confirmed against `conda list -n ark_medmnist` during planning) — leaving that import in place would break every downstream import of this module (`train_ark_18datasets.py`, and `finetune.py`/`linear_probe.py` in the follow-up evaluation plan) with an `ImportError` the moment anyone touched this file again, for a dependency nothing actually needs. `SwinTransformer`/`F` are also dropped: the new code below uses `timm.create_model(...)` (matching the pattern `ArkPlusDual` needs) instead of the raw class, and doesn't need `torch.nn.functional` directly.

Remove `ArkPlus3D` and `ArkPlusStudentTeacher` entirely (the two classes below `MultiTaskHead`). Keep `Projector` and `MultiTaskHead` unchanged (both use `nn`, already imported above). Add, in their place:

```python
class ArkPlusDual(nn.Module):
    """Dual-modality Ark+ student/teacher body (build two instances for the pair).

    Two encoders (2D Swin-Tiny, 3D ResNet-18), each with its own neck into a
    shared_dim, both feeding ONE shared projector -- this is where 2D and 3D
    representations are actually pulled into a common space via the
    consistency loss. Classification heads read the projector's UNNORMALIZED
    output; L2-normalizing before the heads caps softmax/sigmoid logits and
    collapses training to the class prior (~0.15 AUC cost, seen previously).
    """

    def __init__(self, num_classes_list, img_size_2d=112, shared_dim=512,
                 proj_hidden=1024, proj_dim=512, pretrained_2d=True):
        super().__init__()

        self.encoder_2d = timm.create_model(
            "swin_tiny_patch4_window7_224",
            pretrained=pretrained_2d,
            img_size=img_size_2d,
            num_classes=0,
            global_pool="avg",
        )
        self.encoder_3d = ResNet3D18(in_channels=1)

        self.neck_2d = nn.Linear(self.encoder_2d.num_features, shared_dim)
        self.neck_3d = nn.Linear(self.encoder_3d.num_features, shared_dim)

        self.projector = Projector(in_dim=shared_dim, hidden_dim=proj_hidden, out_dim=proj_dim)

        self.heads = nn.ModuleList([
            MultiTaskHead(in_features=proj_dim, num_classes=nc)
            for nc in num_classes_list
        ])

    def _encode(self, x):
        if x.dim() == 5:  # (B, 1, D, H, W) native 3D volume
            feats = self.encoder_3d(x)
            return self.neck_3d(feats)
        if x.shape[1] == 1:  # 2D grayscale -> 3ch for the ImageNet-init encoder
            x = x.expand(-1, 3, -1, -1)
        feats = self.encoder_2d(x)
        return self.neck_2d(feats)

    def forward(self, x, head_n):
        neck_out = self._encode(x)
        proj = self.projector(neck_out)          # UNNORMALIZED — heads + linear probe use this
        logits = self.heads[head_n](proj)
        return proj, logits


def build_student_teacher(num_classes_list, **kwargs):
    student = ArkPlusDual(num_classes_list, **kwargs)
    teacher = ArkPlusDual(num_classes_list, **kwargs)
    teacher.load_state_dict(student.state_dict())
    for p in teacher.parameters():
        p.requires_grad = False
    return student, teacher


@torch.no_grad()
def ema_update(student, teacher, momentum, update_buffers=True):
    """EMA the teacher from the student. Buffer EMA is a correctness fix
    (BatchNorm running stats must track, or cyclic training leaves the
    teacher's running_mean/running_var holding whichever dataset ran last),
    not a fidelity deviation -- kept on by default, flag exists for ablation.
    """
    for sp, tp in zip(student.parameters(), teacher.parameters()):
        tp.data.mul_(momentum).add_(sp.detach().data, alpha=1 - momentum)
    if update_buffers:
        for sb, tb in zip(student.buffers(), teacher.buffers()):
            if tb.dtype.is_floating_point:
                tb.mul_(momentum).add_(sb.detach(), alpha=1 - momentum)
            else:
                tb.copy_(sb)
```

- [ ] **Step 2: Write a self-check**

```python
"""Self-check for ArkPlusDual: run `python test_ark_plus_model.py`."""
import torch
from ark_plus_model import build_student_teacher, ema_update


def test_2d_and_3d_paths_share_head_dim():
    student, teacher = build_student_teacher([9, 2], img_size_2d=112)
    x2d = torch.randn(2, 3, 112, 112)
    x3d = torch.randn(2, 1, 28, 28, 28)

    proj2d, logits2d = student(x2d, head_n=0)
    proj3d, logits3d = student(x3d, head_n=1)

    assert proj2d.shape == proj3d.shape == (2, 512), \
        f"projector output must be shared-shape regardless of modality: {proj2d.shape} vs {proj3d.shape}"
    assert logits2d.shape == (2, 9)
    assert logits3d.shape == (2, 2)


def test_heads_see_unnormalized_projection():
    # If heads accidentally read a normalized vector, scaling the input
    # projection should NOT change the logits (norm-invariant). Heads on the
    # *unnormalized* projection must be scale-sensitive.
    student, _ = build_student_teacher([9])
    proj = torch.randn(4, 512, requires_grad=False)
    logits_a = student.heads[0](proj)
    logits_b = student.heads[0](proj * 10.0)
    assert not torch.allclose(logits_a, logits_b), \
        "heads appear scale-invariant -- are they reading a normalized projection?"


def test_ema_updates_buffers():
    student, teacher = build_student_teacher([9])
    # Force a BatchNorm running_mean to a known nonzero value on the student.
    bn = next(m for m in student.encoder_3d.modules() if isinstance(m, torch.nn.BatchNorm3d))
    bn.running_mean.fill_(5.0)
    ema_update(student, teacher, momentum=0.0, update_buffers=True)  # momentum=0 -> teacher = student
    tbn = next(m for m in teacher.encoder_3d.modules() if isinstance(m, torch.nn.BatchNorm3d))
    assert torch.allclose(tbn.running_mean, torch.full_like(tbn.running_mean, 5.0)), \
        "teacher BatchNorm buffer did not track student under momentum=0"


if __name__ == "__main__":
    test_2d_and_3d_paths_share_head_dim()
    test_heads_see_unnormalized_projection()
    test_ema_updates_buffers()
    print("ark_plus_model.py: all checks passed")
```

- [ ] **Step 3: Run it**

Run: `conda run -n ark_medmnist python test_ark_plus_model.py`
Expected: `ark_plus_model.py: all checks passed`

- [ ] **Step 4: Commit**

```bash
git add ark_plus_model.py test_ark_plus_model.py
git commit -m "fix bug D: add shared projector between dual encoders and heads; heads read unnormalized projection"
```

---

### Task 3: Derive dataset config from medmnist.INFO at runtime (fixes bug B — hardcoded/wrong NUM_CLASSES)

**Files:**
- Create: `dataset_registry.py`
- Test: `test_dataset_registry.py`

**Interfaces:**
- Produces: `DATASETS_2D`, `DATASETS_3D`, `DATASETS` (lists of `medmnist.INFO` keys, e.g. `'pathmnist'`), and `get_dataset_config(key) -> dict` with `num_classes`, `task` (`'multi-class'` or `'multi-label'`), `n_channels`, `python_class`.

This replaces the hardcoded `NUM_CLASSES`/`TASK_TYPE` dicts in `train_ark_18datasets.py`, whose `'BreastMNIST': 3` was wrong (BreastMNIST is 2-class). Verified directly against the installed `medmnist==3.0.2` package during planning: `retinamnist` is `'ordinal-regression'` (5 classes, treated as multi-class/CE here, matching the existing convention already used for the two `'binary-class'` sets); `chestmnist` is `'multi-label, binary-class'` (14 labels); `fracturemnist3d` is 3-class; `synapsemnist3d`/`vesselmnist3d` are both binary.

- [ ] **Step 1: Write the registry**

```python
"""Dataset config derived from medmnist.INFO at import time -- never hardcode
class counts/task types here again (see bug B in the Ark+ replication brief:
a hardcoded 'BreastMNIST': 3 silently corrupted that dataset's AUC)."""
from medmnist import INFO

DATASETS_2D = [
    'pathmnist', 'bloodmnist', 'dermamnist', 'octmnist',
    'pneumoniamnist', 'retinamnist', 'breastmnist', 'tissuemnist',
    'organamnist', 'organcmnist', 'organsmnist', 'chestmnist',
]
DATASETS_3D = [
    'organmnist3d', 'nodulemnist3d', 'adrenalmnist3d',
    'fracturemnist3d', 'vesselmnist3d', 'synapsemnist3d',
]
DATASETS = DATASETS_2D + DATASETS_3D


def get_dataset_config(key):
    info = INFO[key]
    task = 'multi-label' if info['task'].startswith('multi-label') else 'multi-class'
    return {
        'num_classes': len(info['label']),
        'task': task,
        'n_channels': info['n_channels'],
        'python_class': info['python_class'],
    }
```

- [ ] **Step 2: Write a self-check**

```python
"""Self-check for dataset_registry: run `python test_dataset_registry.py`."""
from dataset_registry import DATASETS, DATASETS_2D, DATASETS_3D, get_dataset_config

# Ground truth transcribed from `medmnist.INFO` on the installed medmnist==3.0.2,
# specifically to catch the regressions the brief calls out by name.
EXPECTED = {
    'breastmnist':      {'num_classes': 2,  'task': 'multi-class'},   # was hardcoded 3 -- bug B
    'chestmnist':       {'num_classes': 14, 'task': 'multi-label'},
    'fracturemnist3d':  {'num_classes': 3,  'task': 'multi-class'},   # was documented as binary
    'synapsemnist3d':   {'num_classes': 2,  'task': 'multi-class'},   # binary synapses, not 8-class
    'vesselmnist3d':    {'num_classes': 2,  'task': 'multi-class'},   # binary vessels, not retinal
    'retinamnist':      {'num_classes': 5,  'task': 'multi-class'},   # ordinal-regression -> multi-class
}


def test_known_regressions():
    for key, expected in EXPECTED.items():
        cfg = get_dataset_config(key)
        assert cfg['num_classes'] == expected['num_classes'], \
            f"{key}: expected {expected['num_classes']} classes, got {cfg['num_classes']}"
        assert cfg['task'] == expected['task'], \
            f"{key}: expected task={expected['task']}, got {cfg['task']}"


def test_all_18_resolve():
    assert len(DATASETS) == 18
    assert len(DATASETS_2D) == 12
    assert len(DATASETS_3D) == 6
    for key in DATASETS:
        cfg = get_dataset_config(key)
        assert cfg['num_classes'] > 0
        assert cfg['task'] in ('multi-class', 'multi-label')


if __name__ == "__main__":
    test_known_regressions()
    test_all_18_resolve()
    print("dataset_registry.py: all checks passed")
```

- [ ] **Step 3: Run it**

Run: `conda run -n ark_medmnist python test_dataset_registry.py`
Expected: `dataset_registry.py: all checks passed`

- [ ] **Step 4: Commit**

```bash
git add dataset_registry.py test_dataset_registry.py
git commit -m "fix bug B: derive class counts/task types from medmnist.INFO instead of hardcoded dict"
```

---

### Task 4: Config sizing + dataset wrappers (fixes bug C.3 — symmetric-view bug; wires bug B)

**Files:**
- Modify: `train_ark_18datasets.py` (full rewrite of the CONFIG block, `Dataset2D`, `Dataset3D`)

**Interfaces:**
- Consumes: `dataset_registry.{DATASETS, DATASETS_2D, DATASETS_3D, get_dataset_config}` (Task 3).
- Produces: `Dataset2D(key, split, size)` / `Dataset3D(key, split)` where `__getitem__` returns `(student_view, teacher_view, label)`; `teacher_view` is always the deterministic/un-augmented resize, `student_view` is augmented only when `split == 'train'`.

**Compute budget (must be read before Task 9's run):**

Student ≈ 28M (Swin-Tiny) + 33M (ResNet3D-18) + ~3M (necks/projector/18 heads) ≈ 65M params. Teacher is a second full copy (frozen, no optimizer state). Weights: 2×65M×4B ≈ 520 MB (fp32). SGD momentum buffer + gradients for the student only: 65M×4B×2 ≈ 520 MB. That's ~1.0 GB of static allocation before a single activation is computed. Activations run under AMP (fp16): Swin-Tiny at 112×112 with grad-checkpointing enabled and batch 32 is on the order of ~1-1.5 GB; ResNet3D-18 at native 28³ with batch 8 is tens of MB (channel counts are modest at this tiny spatial resolution, unlike the old slice-folded path which pushed an effective batch of 8×28=224 2D images per 3D step). Rough peak estimate: **3-4.5 GB**, comfortably under the RTX 4060's 8 GB — but only if nothing else is holding VRAM. `nvidia-smi` showed 6.4 GB already in use by desktop/game processes during planning; **close GPU-heavy applications (games especially) before launching training**, or the real headroom is a fraction of the 8 GB total. Treat this estimate as a starting point, not a guarantee — the per-dataset progress printing added in Task 9 reports wall-clock so the true cost is measured on the first cycle, not assumed.

Resolution is **112×112**, not the original script's 64 or the paper's 768: with `patch_size=4, window_size=7`, 112 gives a 28×28 token grid that divides evenly into 4×4 windows (`28/7=4`); 64 gives a 16×16 grid that does *not* divide evenly by 7, forcing timm to pad/mask internally. 112 is simultaneously more correct for the windowing and 4x cheaper than 224. This is a deviation from the paper's 768×768 (documented in `DEVIATIONS.md`, follow-up plan) but the *choice* of 112 over the previous 64 is a straightforward correctness+efficiency improvement, not a fidelity tradeoff.

- [ ] **Step 1: Rewrite the CONFIG block and dataset wrappers**

Replace the entire header (`CONFIG` block, `DATASETS_2D`/`DATASETS_3D`/`DATASETS`, `NUM_CLASSES`, `TASK_TYPE`, `MEDMNIST_2D_MAP`, `_transform`, `Dataset2D`, `MEDMNIST_3D_MAP`, `Dataset3D` — original lines 14–153) with:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import medmnist
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torchvision.transforms as transforms
from sklearn.metrics import roc_auc_score
import os
import time
import sys

from dataset_registry import DATASETS, DATASETS_2D, DATASETS_3D, get_dataset_config
from ark_plus_model import build_student_teacher, ema_update

# ============================================================
# CONFIG — edit these before running. See Task 4 of the core
# training rewrite plan for the compute-budget arithmetic behind
# these numbers; treat them as a starting point, confirm on the
# first cycle's measured wall-clock/VRAM before trusting them.
# ============================================================
EXPERIMENT     = "ark_all_18_datasets_v2"
IMAGE_SIZE     = 112          # DEVIATION from paper's 768 — see DEVIATIONS.md
BATCH_SIZE_2D  = 32
BATCH_SIZE_3D  = 8
EPOCHS         = 25           # DEVIATION from paper's 50 cycles — see DEVIATIONS.md
PATIENCE       = 5
BASE_LR        = 0.2          # DEVIATION: scaled down from paper's 0.3 (tuned for batch 50
                               # across 4 GPUs); linear-scaled here for our much smaller batch.
WARMUP_EPOCHS  = 2             # DEVIATION: paper doesn't warm up at batch 50; small-batch
                               # single-GPU SGD at this LR is unstable without it.
MOMENTUM_EMA   = 0.9
GRAD_CLIP_NORM = 1.0
CHESTMNIST_POS_WEIGHT = False  # DEVIATION if True — not in Ark+ paper; see DEVIATIONS.md
PER_TASK_LOSS_NORM    = False  # DEVIATION if True — not in Ark+ paper; see DEVIATIONS.md
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_DIR       = "./outputs/" + EXPERIMENT
CHECKPOINT      = os.path.join(SAVE_DIR, "checkpoint_latest.pth")
CHECKPOINT_PREV = os.path.join(SAVE_DIR, "checkpoint_prev.pth")
BEST_MODEL      = os.path.join(SAVE_DIR, "best_model.pth")
os.makedirs(SAVE_DIR, exist_ok=True)
# ============================================================


def _transform(size, augment):
    if augment:
        return transforms.Compose([
            transforms.Resize((size, size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


class Dataset2D(Dataset):
    """Returns (student_view, teacher_view, label). Teacher always gets the
    resized-original (no augmentation) per Ark+'s asymmetric-input design --
    the earlier script gave BOTH views the training transform, which meant
    the teacher's "stable supervisory signal" was actually just as noisy as
    the student's (bug C.3)."""

    def __init__(self, key, split, size=IMAGE_SIZE):
        self.key   = key
        self.cfg   = get_dataset_config(key)
        cls        = getattr(medmnist, self.cfg['python_class'])
        self.data  = cls(split=split, download=True, size=size, as_rgb=True)
        self.tf_train = _transform(size, augment=True)
        self.tf_val   = _transform(size, augment=False)
        self.is_train = (split == 'train')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, lbl = self.data[idx]
        if not isinstance(img, Image.Image):
            img = Image.fromarray(np.uint8(img))
        lbl = torch.tensor(lbl).squeeze()
        lbl = lbl.float() if self.cfg['task'] == 'multi-label' else lbl.long()

        student_view = self.tf_train(img) if self.is_train else self.tf_val(img)
        teacher_view = self.tf_val(img)  # always deterministic, even during training
        return student_view, teacher_view, lbl


class Augment3D:
    def __call__(self, vol, augment):
        if augment:
            for ax in range(3):
                if np.random.rand() > 0.5:
                    vol = np.flip(vol, ax).copy()
            vol = np.clip(vol * np.random.uniform(0.85, 1.15)
                          + np.random.uniform(-0.05, 0.05), 0, 1)
        return torch.tensor(vol[None], dtype=torch.float32)  # (1, D, H, W)


class Dataset3D(Dataset):
    def __init__(self, key, split):
        self.key   = key
        self.cfg   = get_dataset_config(key)
        cls        = getattr(medmnist, self.cfg['python_class'])
        raw        = cls(split=split, download=True)
        self.imgs  = raw.imgs.astype(np.float32) / 255.0
        self.lbls  = raw.labels.squeeze().astype(np.int64)
        self.is_train = (split == 'train')
        self.aug = Augment3D()

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        vol = self.imgs[idx]
        student_view = self.aug(vol, augment=self.is_train)
        teacher_view = self.aug(vol, augment=False)
        return student_view, teacher_view, torch.tensor(int(self.lbls[idx]))
```

- [ ] **Step 2: Commit**

```bash
git add train_ark_18datasets.py
git commit -m "fix bug C.3: teacher view is always the deterministic resize, not another augmented crop; size to 8GB budget"
```

---

### Task 5: Wire the dual-encoder model (fixes bug A — slice-averaged 3D path)

**Files:**
- Modify: `train_ark_18datasets.py`

**Interfaces:**
- Consumes: `build_student_teacher` from `ark_plus_model.py` (Task 2).
- Produces: `model`, `teacher` instances used by the rest of `main()`.

- [ ] **Step 1: Remove the slice-averaging model class**

Delete the entire `ArkMedMNIST18` class (original lines 156–197) — the `_encode` method that reshapes `(B,1,D,H,W)` into `(B*D,3,H,W)`, pushes 28 independent slices through the 2D encoder, and averages the results. This is bug A: it destroys inter-slice structure and inflates the effective 3D batch to `BATCH_SIZE_3D × 28`. `ArkPlusDual` (Task 2) replaces it with a native 3D path.

Nothing else needs to change here yet — `main()` still references `ArkMedMNIST18(...)`; that call site is replaced in Task 9 when the full `main()` is reassembled, since it also needs the new dataloaders, checkpoint logic, and training loop from the remaining tasks.

- [ ] **Step 2: Commit**

```bash
git add train_ark_18datasets.py
git commit -m "fix bug A: remove slice-and-average 3D path, replaced by native ArkPlusDual 3D encoder"
```

---

### Task 6: Fix training step — loss combination, ChestMNIST weighting, AMP, gradient clipping (fixes bugs C.1, C.2, D, E, G)

**Files:**
- Modify: `train_ark_18datasets.py`

**Interfaces:**
- Consumes: `ArkPlusDual.forward` returning `(proj, logits)` (Task 2); `CHESTMNIST_POS_WEIGHT`, `PER_TASK_LOSS_NORM`, `GRAD_CLIP_NORM` from Task 4's config block.
- Produces: `train_one_cycle(model, teacher, key, cfg, loader, head_n, optimizer, scaler, device, pos_weight=None) -> (loss, cls, consist)`.

This single function is where bugs D (missing projector — now consumed correctly), E (loss combination), C.1/C.2 (ChestMNIST loss scale/`pos_weight`), and G (no AMP/no grad clip) all live, so they're fixed together rather than as artificial separate diffs against the same 15 lines.

- [ ] **Step 1: Replace `train_one_cycle` and add its loss helpers**

Delete the original `train_one_cycle` (lines 225–243) and `ema_update`/`compute_auc` remain for now (touched in Task 7). Add:

```python
LOSS_SCALE_EMA = {}  # dataset key -> running mean of loss_cls, used only if PER_TASK_LOSS_NORM


def normalize_loss(key, loss_cls, enabled, momentum=0.98):
    """Optional self-normalizing loss: divide by a running EMA of the loss's
    own recent magnitude, so no single dataset's loss scale dominates the
    shared encoder's gradient. DEVIATION, off by default (bug C.1)."""
    if not enabled:
        return loss_cls
    prev = LOSS_SCALE_EMA.get(key, loss_cls.item())
    scale = momentum * prev + (1 - momentum) * loss_cls.item()
    LOSS_SCALE_EMA[key] = scale
    return loss_cls / max(scale, 1e-3)


def make_criterion(cfg, pos_weight=None):
    if cfg['task'] == 'multi-label':
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    return nn.CrossEntropyLoss()


def consistency_loss(proj_s, proj_t):
    """Squared L2 distance between L2-normalized projections, SUMMED over the
    embedding dimension then averaged over the batch -- nn.MSELoss's default
    divides by embedding_dim too, which silently makes a consistency weight
    of 1.0 a near no-op (bug E)."""
    s = F.normalize(proj_s, dim=1)
    t = F.normalize(proj_t, dim=1)
    return (s - t).pow(2).sum(dim=1).mean()


def train_one_cycle(model, teacher, key, cfg, loader, head_n, optimizer, scaler, device,
                     pos_weight=None):
    model.train()
    crit = make_criterion(cfg, pos_weight)
    total_loss = total_cls = total_con = n = 0
    for v1, v2, lbl in loader:
        v1, v2, lbl = v1.to(device), v2.to(device), lbl.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            proj_s, pred_s = model(v1, head_n)
            with torch.no_grad():
                proj_t, _ = teacher(v2, head_n)
            loss_cls = crit(pred_s, lbl)
            loss_con = consistency_loss(proj_s, proj_t)
            loss = normalize_loss(key, loss_cls, PER_TASK_LOSS_NORM) + loss_con  # bug E: L_cls + L_consist, no ramp
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_NORM)  # bug G
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item(); total_cls += loss_cls.item()
        total_con += loss_con.item(); n += 1
    return total_loss / n, total_cls / n, total_con / n
```

- [ ] **Step 2: Commit**

```bash
git add train_ark_18datasets.py
git commit -m "fix bugs D/E/G: heads consume projector output correctly, drop convex loss ramp, add AMP + grad clipping; fix bugs C.1/C.2: gate ChestMNIST loss-scale/pos_weight behind default-off flags"
```

---

### Task 7: Wire AUC computation to dynamic config (fixes bug C.4 — verify evaluation path)

**Files:**
- Modify: `train_ark_18datasets.py`

**Interfaces:**
- Consumes: `get_dataset_config` (Task 3).
- Produces: `compute_auc(y_true, y_pred, cfg) -> float`, `evaluate_auc(model, cfg, loader, head_n, device) -> float`.

The original `evaluate_auc`/`compute_auc` already apply sigmoid for multi-label and softmax for multi-class, and already skip classes absent from a batch (`if oh[:, c].sum() > 0` / `if y_true[:, c].sum() > 0`) — bug C.4 was mostly a false alarm once checked against the actual code, but both functions still read the hardcoded `NUM_CLASSES`/`TASK_TYPE` globals removed in Task 4, so they need updating to take `cfg` instead.

- [ ] **Step 1: Replace both functions**

```python
def compute_auc(y_true, y_pred, cfg):
    y_true = y_true.cpu().numpy()
    y_pred = y_pred.cpu().numpy()
    aucs = []
    if cfg['task'] == 'multi-class':
        oh = np.zeros((len(y_true), cfg['num_classes']))
        for i, v in enumerate(y_true):
            oh[i, int(v)] = 1
        for c in range(cfg['num_classes']):
            if oh[:, c].sum() > 0:
                aucs.append(roc_auc_score(oh[:, c], y_pred[:, c]))
    else:
        for c in range(cfg['num_classes']):
            if y_true[:, c].sum() > 0:
                aucs.append(roc_auc_score(y_true[:, c], y_pred[:, c]))
    return float(np.mean(aucs)) if aucs else 0.0


def evaluate_auc(model, cfg, loader, head_n, device):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for v1, _, lbl in loader:
            v1 = v1.to(device)
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                _, out = model(v1, head_n)
            out = torch.softmax(out, dim=1) if cfg['task'] == 'multi-class' else torch.sigmoid(out)
            preds.append(out.float().cpu()); labels.append(lbl.cpu())
    return compute_auc(torch.cat(labels), torch.cat(preds), cfg)
```

- [ ] **Step 2: Commit**

```bash
git add train_ark_18datasets.py
git commit -m "verify bug C.4: wire AUC computation to dynamic dataset config (sigmoid/softmax + absent-class skip were already correct)"
```

---

### Task 8: Crash-proof checkpoint core — atomic writes, RNG state, fallback (THE MOST IMPORTANT REQUIREMENT)

**Files:**
- Modify: `train_ark_18datasets.py`
- Test: `test_checkpoint.py`

**Interfaces:**
- Produces: `save_checkpoint(state, path_latest, path_prev)`, `load_checkpoint(path_latest, path_prev) -> dict | None`, `capture_rng_state() -> dict`, `restore_rng_state(rng)`.

A plain `torch.save(state, CHECKPOINT)` overwrites the file in place; a power cut mid-write leaves a truncated, unloadable file with no fallback. This writes to a `.tmp` file, `fsync`s it to force it to disk, `os.replace()`s it into place (atomic on both POSIX and Windows — `os.replace` is `MoveFileExW` with `MOVEFILE_REPLACE_EXISTING` on Windows, atomic for same-volume renames), and keeps the previous good checkpoint as a fallback.

- [ ] **Step 1: Write the checkpoint functions**

```python
def save_checkpoint(state, path_latest, path_prev):
    """Atomic checkpoint write: tmp file -> fsync -> os.replace(). A power
    cut can only ever leave the OLD checkpoint_latest.pth intact (os.replace
    either fully lands or doesn't happen at all -- never a half-written
    file) or, at worst, an orphaned .tmp file that the next save overwrites.
    """
    tmp_path = path_latest + ".tmp"
    with open(tmp_path, "wb") as f:
        torch.save(state, f)
        f.flush()
        os.fsync(f.fileno())
    if os.path.isfile(path_latest):
        os.replace(path_latest, path_prev)
    os.replace(tmp_path, path_latest)


def load_checkpoint(path_latest, path_prev):
    """Try the latest checkpoint; if it's missing or fails to load (e.g. a
    prior run was killed between the two os.replace() calls, or the disk
    itself corrupted the file), fall back to the previous one."""
    for path, tag in [(path_latest, "latest"), (path_prev, "previous")]:
        if os.path.isfile(path):
            try:
                ckpt = torch.load(path, map_location="cpu", weights_only=False)
                print(f">>> Loaded {tag} checkpoint: {path}")
                return ckpt
            except Exception as e:
                print(f"WARNING: failed to load {tag} checkpoint '{path}': {e}")
    return None


def capture_rng_state():
    return {
        'torch': torch.get_rng_state(),
        'numpy': np.random.get_state(),
        'python': random.getstate(),
        'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(rng):
    torch.set_rng_state(rng['torch'])
    np.random.set_state(rng['numpy'])
    random.setstate(rng['python'])
    if rng['cuda'] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(rng['cuda'])
```

- [ ] **Step 2: Write a self-check covering the atomic-write and fallback paths**

```python
"""Self-check for checkpoint save/load: run `python test_checkpoint.py`.
Covers the two failure modes the brief calls out explicitly: a truncated
'latest' file (simulating a power cut mid-write) and full RNG round-trip.
"""
import os
import random
import shutil
import tempfile

import numpy as np
import torch

from train_ark_18datasets import (
    save_checkpoint, load_checkpoint, capture_rng_state, restore_rng_state,
)


def test_round_trip():
    d = tempfile.mkdtemp()
    try:
        latest, prev = os.path.join(d, "latest.pth"), os.path.join(d, "prev.pth")
        state = {"epoch": 3, "dataset_index": 7, "weights": torch.randn(10)}
        save_checkpoint(state, latest, prev)
        loaded = load_checkpoint(latest, prev)
        assert loaded["epoch"] == 3 and loaded["dataset_index"] == 7
        assert torch.allclose(loaded["weights"], state["weights"])
    finally:
        shutil.rmtree(d)


def test_falls_back_to_prev_when_latest_is_corrupt():
    d = tempfile.mkdtemp()
    try:
        latest, prev = os.path.join(d, "latest.pth"), os.path.join(d, "prev.pth")
        save_checkpoint({"epoch": 1, "dataset_index": 0}, latest, prev)
        save_checkpoint({"epoch": 2, "dataset_index": 0}, latest, prev)  # prev now holds epoch=1
        # Simulate a power cut mid-write: truncate 'latest' to garbage bytes.
        with open(latest, "wb") as f:
            f.write(b"not a valid checkpoint")
        loaded = load_checkpoint(latest, prev)
        assert loaded is not None, "should have fallen back to prev, not returned None"
        assert loaded["epoch"] == 1, f"expected fallback to prev (epoch=1), got {loaded['epoch']}"
    finally:
        shutil.rmtree(d)


def test_rng_round_trip_reproduces_next_draws():
    rng = capture_rng_state()
    expected_torch = torch.rand(5)
    expected_np = np.random.rand(5)
    expected_py = [random.random() for _ in range(5)]

    torch.rand(100); np.random.rand(100); [random.random() for _ in range(100)]  # perturb state

    restore_rng_state(rng)
    assert torch.allclose(torch.rand(5), expected_torch)
    assert np.allclose(np.random.rand(5), expected_np)
    assert [random.random() for _ in range(5)] == expected_py


if __name__ == "__main__":
    test_round_trip()
    test_falls_back_to_prev_when_latest_is_corrupt()
    test_rng_round_trip_reproduces_next_draws()
    print("test_checkpoint.py: all checks passed")
```

`test_checkpoint.py` imports `train_ark_18datasets`, which at import time builds `SAVE_DIR` and sets up config — that's safe (no training starts on import; `main()` only runs under `if __name__ == "__main__"`, added in Task 9).

- [ ] **Step 3: Run it**

Run: `conda run -n ark_medmnist python test_checkpoint.py`
Expected: `test_checkpoint.py: all checks passed`

- [ ] **Step 4: Commit**

```bash
git add train_ark_18datasets.py test_checkpoint.py
git commit -m "add crash-proof checkpointing: atomic writes, RNG state capture, corrupt-file fallback to prev"
```

---

### Task 9: Resumable main loop — per-dataset checkpointing, KeyboardInterrupt handling, append-mode logging, progress/ETA

**Files:**
- Modify: `train_ark_18datasets.py`

**Interfaces:**
- Consumes: everything from Tasks 4–8.
- Produces: the final `main()`, run via `python train_ark_18datasets.py`.

This is where "checkpoint after every dataset, not every epoch" actually happens. Key design point: track `last_completed` (the last dataset index that **fully finished** training + EMA this epoch, `-1` if none yet) separately from the loop variable `i`. If `KeyboardInterrupt` fires mid-`train_one_cycle`, `i` is *in progress*, not done — checkpointing with `dataset_index=i` would wrongly mark it complete and resume would skip it, silently losing that dataset's exposure for the epoch. Checkpointing `last_completed` instead means resume always redoes at most one partially-trained dataset, never skips one.

Resume position falls out of one loop bound with no special-casing: `first_i = last_completed + 1` only on the epoch matching `start_epoch`, else `0`. If a checkpoint was saved right after the *last* dataset of an epoch finished (before validation ran), `first_i` naturally equals `N`, `range(N, N)` is empty, and control falls straight through to validation — no separate "all datasets done, advance epoch" branch needed.

- [ ] **Step 1: Replace the entire `main()` function and the trailing `main()` call**

```python
def build_loaders():
    train_loaders, val_loaders, test_loaders, cfgs, pos_weights = {}, {}, {}, {}, {}
    for key in DATASETS:
        is3d = key in DATASETS_3D
        Cls = Dataset3D if is3d else Dataset2D
        cfg = get_dataset_config(key)
        cfgs[key] = cfg

        train_ds = Cls(key, 'train') if is3d else Cls(key, 'train', IMAGE_SIZE)
        val_ds   = Cls(key, 'val')   if is3d else Cls(key, 'val',   IMAGE_SIZE)
        test_ds  = Cls(key, 'test')  if is3d else Cls(key, 'test',  IMAGE_SIZE)

        bs = BATCH_SIZE_3D if is3d else BATCH_SIZE_2D
        train_loaders[key] = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=0, pin_memory=True)
        val_loaders[key]   = DataLoader(val_ds,   batch_size=bs, shuffle=False, num_workers=0, pin_memory=True)
        test_loaders[key]  = DataLoader(test_ds,  batch_size=bs, shuffle=False, num_workers=0, pin_memory=True)
        print(f"  {key}: train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")

        if key == 'chestmnist' and CHESTMNIST_POS_WEIGHT:
            rates = torch.tensor(train_ds.data.labels.mean(axis=0), dtype=torch.float32).clamp(1e-3, 1 - 1e-3)
            pos_weights[key] = ((1 - rates) / rates).to(DEVICE)
        else:
            pos_weights[key] = None
    return train_loaders, val_loaders, test_loaders, cfgs, pos_weights


def make_optimizer_and_scheduler(model):
    optimizer = torch.optim.SGD(model.parameters(), lr=BASE_LR, momentum=0.9, weight_decay=1e-4)
    warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=WARMUP_EPOCHS)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(EPOCHS - WARMUP_EPOCHS, 1))
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[WARMUP_EPOCHS])
    return optimizer, scheduler


def main():
    print("=" * 60)
    print("EXPERIMENT :", EXPERIMENT)
    print("DATASETS   :", len(DATASETS_2D), "x 2D +", len(DATASETS_3D), "x 3D =", len(DATASETS), "total")
    print("EPOCHS     :", EPOCHS, "  PATIENCE:", PATIENCE, "  IMAGE_SIZE:", IMAGE_SIZE)
    print("DEVICE     :", DEVICE)
    print("=" * 60)

    print("\nBuilding dataloaders...")
    train_loaders, val_loaders, test_loaders, cfgs, pos_weights = build_loaders()

    print("\nBuilding model...")
    num_classes_list = [cfgs[k]['num_classes'] for k in DATASETS]
    model, teacher = build_student_teacher(num_classes_list, img_size_2d=IMAGE_SIZE)
    model.to(DEVICE); teacher.to(DEVICE)
    model.encoder_2d.set_grad_checkpointing(True)
    print(f"  Student params: {sum(p.numel() for p in model.parameters())//1_000_000}M")

    optimizer, scheduler = make_optimizer_and_scheduler(model)
    scaler = torch.amp.GradScaler('cuda')  # torch.cuda.amp.GradScaler() is deprecated on torch 2.13

    log_path = os.path.join(SAVE_DIR, "train_log.txt")
    ckpt = load_checkpoint(CHECKPOINT, CHECKPOINT_PREV)

    if ckpt is not None:
        model.load_state_dict(ckpt['student'])
        teacher.load_state_dict(ckpt['teacher'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        scaler.load_state_dict(ckpt['scaler'])
        restore_rng_state(ckpt['rng'])
        best_avg_auc = ckpt['best_avg_auc']
        patience_ctr = ckpt['patience_ctr']
        history = ckpt['history']
        start_epoch = ckpt['epoch']
        last_completed = ckpt['dataset_index']
        skipped = min(last_completed + 1, len(DATASETS))
        print(f">>> Resuming at epoch {start_epoch + 1}/{EPOCHS}, skipping {skipped}/{len(DATASETS)} "
              f"already-completed datasets this epoch: {DATASETS[:skipped]}")
        log_mode = "a"
    else:
        best_avg_auc, patience_ctr, history = 0.0, 0, []
        start_epoch, last_completed = 0, -1
        print(">>> No checkpoint found — starting fresh")
        log_mode = "w"

    with open(log_path, log_mode) as f:
        if log_mode == "w":
            f.write(f"Experiment: {EXPERIMENT}\nDatasets: {DATASETS}\nEpochs: {EPOCHS}  Patience: {PATIENCE}\n\n")
        f.flush()

    def checkpoint_now(epoch, dataset_index):
        state = {
            'epoch': epoch, 'dataset_index': dataset_index,
            'student': model.state_dict(), 'teacher': teacher.state_dict(),
            'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
            'scaler': scaler.state_dict(),
            'best_avg_auc': best_avg_auc, 'patience_ctr': patience_ctr,
            'history': history, 'rng': capture_rng_state(), 'datasets': DATASETS,
        }
        save_checkpoint(state, CHECKPOINT, CHECKPOINT_PREV)

    print("\nStarting cyclic pretraining...")
    stopped_epoch = EPOCHS
    try:
        for epoch in range(start_epoch, EPOCHS):
            first_i = (last_completed + 1) if epoch == start_epoch else 0
            if first_i > 0:
                print(f"  Resuming epoch {epoch + 1}: skipping datasets 0..{first_i - 1}")

            n_remaining = len(DATASETS) - first_i
            t_epoch = time.time()
            for pos, i in enumerate(range(first_i, len(DATASETS))):
                key = DATASETS[i]
                t0 = time.time()
                loss, cls_l, con_l = train_one_cycle(
                    model, teacher, key, cfgs[key], train_loaders[key], i,
                    optimizer, scaler, DEVICE, pos_weight=pos_weights[key])
                ema_update(model, teacher, MOMENTUM_EMA, update_buffers=True)
                last_completed = i

                dt = time.time() - t0
                remaining = n_remaining - pos - 1
                eta_min = (time.time() - t_epoch) / (pos + 1) * remaining / 60
                tag = "3D" if key in DATASETS_3D else "2D"
                print(f"  [{tag}] {key}: loss={loss:.4f} cls={cls_l:.4f} consist={con_l:.4f} "
                      f"({dt:.1f}s, ETA {eta_min:.1f} min for rest of epoch)")

                checkpoint_now(epoch, last_completed)

            scheduler.step()
            last_completed = -1  # next epoch starts clean

            print("\n  Validation AUC:")
            auc_list = []
            for key in DATASETS:
                auc = evaluate_auc(teacher, cfgs[key], val_loaders[key], DATASETS.index(key), DEVICE)
                auc_list.append(auc)
                tag = "3D" if key in DATASETS_3D else "2D"
                print(f"    [{tag}] {key}: {auc:.4f}")

            avg_auc = float(np.mean(auc_list))
            auc_2d = float(np.mean([a for a, k in zip(auc_list, DATASETS) if k in DATASETS_2D]))
            auc_3d = float(np.mean([a for a, k in zip(auc_list, DATASETS) if k in DATASETS_3D]))
            epoch_time = time.time() - t_epoch
            print(f"    --- 2D avg: {auc_2d:.4f} | 3D avg: {auc_3d:.4f} | Overall: {avg_auc:.4f} ---")
            print(f"  Epoch time: {epoch_time / 60:.1f} min")

            history.append({'epoch': epoch, 'avg_auc': avg_auc, 'auc_2d': auc_2d, 'auc_3d': auc_3d,
                             'auc_list': auc_list, 'epoch_time_min': epoch_time / 60})
            with open(log_path, "a") as f:
                f.write(f"Epoch {epoch+1}: avg={avg_auc:.4f} 2D={auc_2d:.4f} 3D={auc_3d:.4f} time={epoch_time/60:.1f}min\n")
                for key, auc in zip(DATASETS, auc_list):
                    f.write(f"  {key}: {auc:.4f}\n")
                f.write("\n"); f.flush()

            checkpoint_now(epoch, last_completed)  # epoch fully done, dataset_index=-1

            if avg_auc > best_avg_auc:
                best_avg_auc, patience_ctr = avg_auc, 0
                save_checkpoint({'epoch': epoch, 'teacher': teacher.state_dict(), 'avg_auc': avg_auc,
                                  'auc_2d': auc_2d, 'auc_3d': auc_3d, 'auc_list': auc_list,
                                  'datasets': DATASETS}, BEST_MODEL, BEST_MODEL + ".prev")
                print(f"  New best saved: overall={avg_auc:.4f}")
            else:
                patience_ctr += 1
                print(f"  No improvement. Patience: {patience_ctr}/{PATIENCE}")
                if patience_ctr >= PATIENCE:
                    stopped_epoch = epoch + 1
                    print(f"\n  EARLY STOPPING at epoch {stopped_epoch}")
                    with open(log_path, "a") as f:
                        f.write(f"Early stopping at epoch {stopped_epoch}\n")
                    break
    except KeyboardInterrupt:
        print("\n>>> KeyboardInterrupt caught — checkpointing before exit...")
        checkpoint_now(epoch, last_completed)
        print(f">>> Saved checkpoint at epoch={epoch}, dataset_index={last_completed}. Exiting.")
        sys.exit(1)

    print(f"\nDone. Best val AUC: {best_avg_auc:.4f}. Stopped at epoch {stopped_epoch}.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run every self-test once more against the fully assembled file**

Run: `conda run -n ark_medmnist python test_checkpoint.py && conda run -n ark_medmnist python test_ark_plus_model.py && conda run -n ark_medmnist python test_resnet3d.py && conda run -n ark_medmnist python test_dataset_registry.py`
Expected: all four print their `all checks passed` line.

- [ ] **Step 3: Commit**

```bash
git add train_ark_18datasets.py
git commit -m "implement deliverable 2: per-dataset crash-proof resume with atomic checkpoints, KeyboardInterrupt handling, append-mode logging, per-dataset ETA"
```

---

### Task 10: Manual crash-resume verification (the brief's explicit test requirement)

**Files:** none (manual procedure — this exercises the running process, not something a unit test can cover).

The brief requires proving resume actually works under both a clean interrupt and a hard kill, not just that the code compiles:

- [ ] **Step 1: Start training, let it run past a couple datasets**

Run: `conda run -n ark_medmnist python train_ark_18datasets.py`

Watch the console until at least 2-3 `[2D] <name>: loss=...` lines have printed (confirms at least one `checkpoint_now` call has landed), then `Ctrl+C`.

Expected: `>>> KeyboardInterrupt caught — checkpointing before exit...` followed by `>>> Saved checkpoint at epoch=0, dataset_index=<N-1>. Exiting.` where `dataset_index` matches the last dataset that finished printing.

- [ ] **Step 2: Restart and confirm it resumes mid-cycle, not from scratch**

Run: `conda run -n ark_medmnist python train_ark_18datasets.py`

Expected: `>>> Loaded latest checkpoint: ...` then `>>> Resuming at epoch 1/25, skipping N already-completed datasets this epoch: [...]`, and the first dataset trained is the one immediately *after* the last one seen in Step 1 — not `pathmnist` again.

- [ ] **Step 3: Repeat with a hard kill (no graceful shutdown path)**

Let it run past 2-3 more datasets, then from a separate terminal: `taskkill /F /PID <pid>` (find the PID via `Get-Process python` in PowerShell or Task Manager). This bypasses the `KeyboardInterrupt` handler entirely, exercising only the periodic `checkpoint_now()` calls and the atomic-write/fallback path for real (not simulated, unlike Task 8's test).

Restart the script again. Expected: resumes from whichever dataset last completed *before* the kill (it may redo the dataset that was in-flight at kill time — that's correct, not a bug: partial progress *within* a single dataset's epoch is not checkpointed, only whole-dataset completions are, per the brief's required granularity).

- [ ] **Step 4: Confirm the loss curve is continuous, not restarted**

Open `outputs/<EXPERIMENT>/train_log.txt`. Confirm it was appended to (not truncated) across both restarts — the log should contain one continuous sequence of epoch entries reflecting all run segments, not just the last one.

---

## Self-Review Notes

- **Spec coverage:** bug A (Task 5, native 3D via Task 2's `ArkPlusDual`), bug B (Task 3), bug C.1/C.2 (Task 6, flag-gated), bug C.3 (Task 4), bug C.4 (Task 7, was already correct), bug D (Task 2), bug E (Task 6), bug F (Task 2's `ema_update`), bug G (Task 6 loss/AMP + Task 9 progress/ETA). Deliverable 2 (crash-proof resume) — Tasks 8-10. Deliverable 3 (bug fixes, each its own commit) — Tasks 1-9, each commit message names its bug letter(s).
- **Not yet covered by this plan, deliberately:** `DEVIATIONS.md` and the results/evaluation harness (deliverables 4-5) — separate plan, since they depend on this script actually training and producing a checkpoint first.
- **No placeholders:** every step has complete, runnable code; no TODOs.
- **Type consistency check:** `ArkPlusDual.forward` returns `(proj, logits)` consistently across Task 2 (definition), Task 6 (`train_one_cycle` unpacks `proj_s, pred_s` / `proj_t, _`), and Task 7 (`evaluate_auc` unpacks `_, out`). `get_dataset_config(key)` dict shape (`num_classes`, `task`, `n_channels`, `python_class`) is consistent across Tasks 3, 4, 6, 7, 9. Checkpoint dict keys (`epoch`, `dataset_index`, `student`, `teacher`, `optimizer`, `scheduler`, `scaler`, `best_avg_auc`, `patience_ctr`, `history`, `rng`, `datasets`) match between `checkpoint_now` (write, Task 9) and the resume block (read, Task 9) and `test_checkpoint.py` (Task 8, uses a smaller ad-hoc dict but exercises the same `save_checkpoint`/`load_checkpoint` functions, not the schema).

---
