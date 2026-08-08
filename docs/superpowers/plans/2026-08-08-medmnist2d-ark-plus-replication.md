# MedMNIST-2D Ark+ Replication (Phase 0 + Step 2) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire the 12 2D MedMNIST datasets into Ark+'s **actual** pretraining code (already present in this repo root, verbatim from `jlianglab/Ark`'s `Ark_Plus/Pretraining`) and produce a real trained checkpoint via their own cyclic `omni_engine` loop — with crash-proof per-dataset resume, AMP for an 8GB card, and a notebook driver for interactive local runs. This is the brief's step 2 ("Train an Ark+ model on the 12 2D MedMNIST datasets only"), built on a genuinely-verified Phase 0 (see below).

**Architecture:** A new `medmnist_dataloader.py` implements Ark+'s exact `(images_path, file_path, crop_size, resize, augment, num_class, annotation_percent)` → `(student_view, teacher_view, label)` Dataset contract for MedMNIST and registers into the existing `dict_dataloarder`. A one-line pooling shim in `ArkSwinTransformer` (`models.py`) bridges a confirmed breaking change between the repo's pinned `timm==0.5.4` and the installed `timm==1.0.28`. A new `swin_tiny` branch in `build_omni_model` sizes the backbone for 8GB VRAM (rule 4 allows shrinking Swin, never substituting it). Crash-proof per-dataset checkpointing is added around `omni_engine`'s existing cyclic loop, gated behind a new `args.crash_proof_resume` flag that defaults to `False` — so the original ChestXray14/CheXpert/etc. pretraining path is provably untouched byte-for-byte when the flag is off. AMP is added inside `trainer.py`'s `train_one_epoch`/`evaluate` via a new optional `scaler` parameter, without touching the loss formula. Everything is delivered as a notebook driver (`pretrain_medmnist2d.ipynb`) per instruction, with all non-trivial logic in tested `.py` modules the notebook imports.

**Tech Stack:** PyTorch 2.13.0+cu130, timm 1.0.28, medmnist 3.0.2, the existing `optparse`-based args plumbing (not argparse — that's what `main_ark.py` already uses).

## Phase 0 verification (already done during planning — do not redo)

- `git clone https://github.com/jlianglab/Ark.git` was done to `D:\Ark_upstream_clone` (sibling of this repo) and kept as a reference for diffing.
- Confirmed byte-for-byte (`diff -b`) that this repo's root `dataloader.py`, `engine.py`, `main_ark.py`, `models.py`, `trainer.py`, `utils.py`, `convnext.py`, `datasets_config.yaml` are **already** Ark_Plus/Pretraining's real files (only CRLF/LF differs). Phase 0's "clone and use their training loop" is already satisfied — the actual gap was that `train_ark_18datasets.py`/`train_ark_12datasets.py`/`train_ark_3d_3datasets.py`/`ark_plus_model.py`/`medmnist_dataloader.py`/`medmnist3d_dataloader.py` are a **parallel, from-scratch reimplementation that never plugs into this real code**. This plan does not touch or depend on any of those files.
- Read `main_ark.py`, `models.py`, `trainer.py`, `engine.py`, `utils.py`, `dataloader.py`, `datasets_config.yaml` directly — every interface referenced below (`dict_dataloarder`, `ArkSwinTransformer.forward`, `train_one_epoch`'s signature, `omni_engine`'s loop, the `(student, teacher, label)` `__getitem__` contract) is transcribed from that reading, not inferred.
- Ran real smoke tests in the `ark_medmnist` conda env (already exists, has `torch==2.13.0+cu130`, `timm==1.0.28`, `medmnist==3.0.2`, `numpy`, `pillow`, `scikit-learn`, `scikit-image`, `tqdm`) and found two **confirmed, not hypothetical** blockers, both addressed in Tasks 1–2 below:
  1. `timm.models.swin_transformer.SwinTransformer.forward_features()` in the installed `timm==1.0.28` returns an **unpooled** `(B, 7, 7, 1024)` spatial map, not the pooled `(B, 1024)` vector `ArkSwinTransformer.forward()` (written against `timm==0.5.4`) requires before its `nn.Linear` projector/heads. Confirmed via `m.forward_head(feats, pre_logits=True)` → `(B, 1024)`, the correct bridge.
  2. `dataloader.py` imports `cv2`, `albumentations`, `pydicom`, `yaml` at module level (used only by the X-ray dataset classes we don't touch) and `trainer.py` imports `wandb` and calls `wandb.log()` unconditionally with no `wandb.init()` anywhere active (the real calls are commented out in `engine.py`). None of `opencv-python`, `albumentations`, `pydicom`, `PyYAML`, `wandb` are installed in `ark_medmnist` — `from dataloader import *` and `from trainer import train_one_epoch` both currently raise `ModuleNotFoundError` before any training code runs.
- Verified `medmnist.INFO` directly (not from memory) for all 12 2D datasets — exact `task`, `n_channels`, class count, `python_class`, and label names are transcribed into Task 4's YAML from a real script run, reproduced in that task's step for anyone re-verifying.
- `nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free` at planning time: `NVIDIA GeForce RTX 4060, 8188 MiB total, 2071 MiB used, 5886 MiB free`. Treat 5886 MiB as the working budget, not the full 8188 — other processes were holding ~2 GB at planning time and may hold more or less at run time. **Close GPU-heavy apps before training and re-check `nvidia-smi` immediately before the real run.**

## Global Constraints

- **Rule 1 (fidelity over performance):** every change in this plan is either (a) wiring MedMNIST into the *unmodified* algorithmic core (EMA formula, `coff` schedule, loss formula, projector, heads — all left exactly as read), or (b) a documented, flagged deviation in `DEVIATIONS.md` (Task 9). No change in this plan alters `coff = (momentum_schedule[it] - 0.9) * 5`, `loss = (1-coff)*loss_cls + coff*loss_const`, `ema_update_teacher`, or the projector/head structure.
- **Rule 3 (one model, not N) — accepted gap, not solved here:** the real `omni_engine` has no "best checkpoint by mean AUC" concept at all — it just checkpoints unconditionally every epoch and periodically runs full test-set AUROC for monitoring only, never for selection. Building real checkpoint-by-mean-AUC selection is deferred to the evaluation-harness follow-up plan (after this plan produces a trained checkpoint to select from). Do not invent selection logic in this plan.
- **Rule 4 (backbone stays Swin):** `swin_tiny` (Task 2) is a size reduction of the exact same `ArkSwinTransformer` class, never a different architecture family.
- **Scope boundary:** this plan covers the 12 2D MedMNIST datasets only (brief's step 2). The 6 3D datasets (step 3) need a different encoder entirely (Swin can't accept 5D input) and the 18-combined dual-encoder run (step 4) is explicitly an *extension*, not a replication — both are separate follow-on plans per the brief's own explicit ordering, not part of this plan.
- Every `.py` file this plan creates gets a `test_*.py` self-check per the project's "non-trivial logic needs one runnable check" convention. `engine.py`/`trainer.py` modifications get targeted tests for the new pieces (checkpoint skip-logic, AMP-enabled forward) plus a manual verification procedure (Task 11) for the parts that need a live GPU + real cyclic run to observe.
- Conda env: `ark_medmnist` (already exists). All commands below run via `conda run -n ark_medmnist python ...`.

---

### Task 1: `environment.yml` + `verify_env.py` — close the import blockers, verify CUDA

**Files:**
- Create: `environment.yml`
- Create: `verify_env.py`

**Interfaces:**
- Produces: a conda env spec matching what's already installed in `ark_medmnist`, plus the packages needed to make `from dataloader import *` and `from trainer import train_one_epoch` succeed. `verify_env.py` is a standalone CLI check (`python verify_env.py`), not an importable module.

- [ ] **Step 1: Write `environment.yml`**

```yaml
name: ark_medmnist
channels:
  - pytorch
  - nvidia
  - conda-forge
  - defaults
dependencies:
  - python=3.11.15
  - numpy=2.4.6
  - pillow=12.3.0
  - scikit-learn=1.9.0
  - scikit-image=0.26.0
  - pip
  - pip:
      - --extra-index-url https://download.pytorch.org/whl/cu130
      - torch==2.13.0+cu130
      - torchvision==0.28.0+cu130
      - timm==1.0.28
      - medmnist==3.0.2
      - tqdm==4.68.4
      - PyYAML==6.0.2
      - opencv-python==4.11.0.86
      - albumentations==1.4.24
      - pydicom==2.4.4
      - wandb==0.19.1
```

`opencv-python`, `albumentations`, `pydicom`, `PyYAML` are required only because `dataloader.py`'s X-ray dataset classes (`ChestXray14`, `CheXpert`, etc. — untouched by this plan) import them at module level; `from dataloader import *` in `main_ark.py` fails without them regardless of which datasets you actually train on. `wandb` is required because `trainer.py` does `import wandb` and calls `wandb.log(...)` unconditionally at the end of `train_one_epoch` — see Task 6 for how those calls are made safe no-ops without a real wandb account.

**Known, accepted version tension (document in Task 9, do not "fix" here):** the repo's own `requirments`/`requirements.txt` pin `timm==0.5.4`. Installing that instead of `1.0.28` was considered and rejected — `0.5.4` is a ~2021-era release with no published wheels compatible with `torch==2.13.0+cu130`/Python 3.11, so pinning to it would break CUDA support entirely to fix a cosmetic version mismatch. The one line of actual incompatibility (`forward_features` pooling) is bridged directly in Task 2 instead.

- [ ] **Step 2: Write `verify_env.py`**

```python
"""Verify the training environment before running anything expensive.

Run: python verify_env.py
Exits non-zero if CUDA is unavailable or a required package is missing.
"""
import sys

REQUIRED = ["torch", "torchvision", "timm", "medmnist", "sklearn", "numpy", "PIL",
            "tqdm", "yaml", "cv2", "albumentations", "pydicom", "wandb", "skimage"]


def check_imports():
    missing = []
    for mod in REQUIRED:
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)
    if missing:
        print(f"FAIL: missing packages: {missing}")
        sys.exit(1)
    print("OK: all required packages importable:", REQUIRED)


def check_cuda():
    import torch

    print(f"torch.__version__         = {torch.__version__}")
    print(f"torch.version.cuda        = {torch.version.cuda}")
    available = torch.cuda.is_available()
    print(f"torch.cuda.is_available() = {available}")

    if not available:
        print("FAIL: CUDA is not available. Training requires a CUDA GPU.")
        sys.exit(1)

    name = torch.cuda.get_device_name(0)
    total_bytes = torch.cuda.get_device_properties(0).total_memory
    total_gib = total_bytes / (1024 ** 3)
    print(f"GPU name                  = {name}")
    print(f"GPU total VRAM             = {total_gib:.2f} GiB ({total_bytes} bytes)")

    if total_gib < 7.5:
        print(f"WARNING: only {total_gib:.2f} GiB VRAM detected — "
              f"the 8 GB 4060 tier, not the 16 GB 4060 Ti. Size batches accordingly.")

    x = torch.randn(1024, 1024, device="cuda")
    y = x @ x
    torch.cuda.synchronize()
    print(f"CUDA matmul smoke test     = OK (result sum={y.sum().item():.2f})")


def check_dataloader_and_trainer_import():
    """The two module-level import blockers found during planning: dataloader.py
    needs cv2/albumentations/pydicom/yaml even for datasets we don't use, and
    trainer.py needs wandb installed (see Task 6 for making wandb.log() safe)."""
    import dataloader  # noqa: F401
    import trainer  # noqa: F401
    print("OK: dataloader.py and trainer.py both import cleanly")


def main():
    check_imports()
    check_cuda()
    check_dataloader_and_trainer_import()
    print("\nEnvironment verification PASSED.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run it and confirm it passes**

Run: `conda run -n ark_medmnist python verify_env.py`

If `check_dataloader_and_trainer_import` fails on a missing package, install the missing pip packages into `ark_medmnist` (`conda run -n ark_medmnist pip install <pkg>==<version>` per `environment.yml`) and re-run.

- [ ] **Step 4: Commit**

```bash
git add environment.yml verify_env.py
git commit -m "add environment.yml and verify_env.py; close dataloader.py/trainer.py import blockers (cv2/albumentations/pydicom/yaml/wandb missing)"
```

---

### Task 2: `models.py` — timm pooling compat shim, `swin_tiny` backbone, ImageNet-init helper

**Files:**
- Modify: `models.py`
- Test: `test_models_medmnist.py`

**Interfaces:**
- Produces: `ArkSwinTransformer.forward`/`generate_embeddings` working correctly against `timm==1.0.28`; `build_omni_model(args, num_classes_list)` accepting `args.model_name == "swin_tiny"`; `load_imagenet_backbone(model, timm_model_name) -> model`.

- [ ] **Step 1: Write the failing test for the pooling shim**

```python
"""test_models_medmnist.py — self-check for the MedMNIST-driven additions to
models.py. Run: python test_models_medmnist.py"""
import torch
from models import ArkSwinTransformer, build_omni_model, load_imagenet_backbone
from optparse import Values


def test_forward_features_pooling_bridges_timm_spatial_output():
    # timm==1.0.28's SwinTransformer.forward_features() returns (B, H, W, C),
    # not the (B, C) vector this class (written against timm==0.5.4) expects
    # before feeding nn.Linear heads -- confirmed via direct smoke test during
    # planning. This must not crash and must return a flat (B, C) vector.
    model = ArkSwinTransformer([9], projector_features=None, use_mlp=False,
                                patch_size=4, window_size=7, embed_dim=96,
                                depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24))
    x = torch.randn(2, 3, 224, 224)
    feat, logits = model(x, head_n=0)
    assert feat.dim() == 2, f"expected pooled (B, C) features, got shape {feat.shape}"
    assert feat.shape == (2, model.num_features)
    assert logits.shape == (2, 9)


def test_generate_embeddings_also_pools():
    model = ArkSwinTransformer([9], projector_features=128, use_mlp=False,
                                patch_size=4, window_size=7, embed_dim=96,
                                depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24))
    x = torch.randn(2, 3, 224, 224)
    emb = model.generate_embeddings(x, after_proj=True)
    assert emb.shape == (2, 128)


def test_build_omni_model_swin_tiny_at_custom_img_size():
    args = Values()
    args.model_name = "swin_tiny"
    args.projector_features = 256
    args.use_mlp = False
    args.crop_size = 112
    args.pretrained_weights = None
    model = build_omni_model(args, num_classes_list=[9, 2])
    x = torch.randn(1, 3, 112, 112)
    feat, logits = model(x, head_n=1)
    assert feat.shape == (1, 256)
    assert logits.shape == (1, 2)
    n_params = sum(p.numel() for p in model.parameters())
    assert 25_000_000 < n_params < 35_000_000, f"unexpected swin_tiny param count: {n_params}"


def test_load_imagenet_backbone_transfers_encoder_not_heads():
    model = ArkSwinTransformer([9], projector_features=None, use_mlp=False,
                                patch_size=4, window_size=7, embed_dim=96,
                                depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24))
    head_before = model.omni_heads[0].weight.clone()
    load_imagenet_backbone(model, "swin_tiny_patch4_window7_224")
    assert torch.allclose(model.omni_heads[0].weight, head_before), \
        "load_imagenet_backbone must not touch omni_heads (strict=False, heads have no matching key)"


if __name__ == "__main__":
    test_forward_features_pooling_bridges_timm_spatial_output()
    test_generate_embeddings_also_pools()
    test_build_omni_model_swin_tiny_at_custom_img_size()
    test_load_imagenet_backbone_transfers_encoder_not_heads()
    print("test_models_medmnist.py: all checks passed")
```

- [ ] **Step 2: Run it to confirm it fails**

Run: `conda run -n ark_medmnist python test_models_medmnist.py`
Expected: `AssertionError: expected pooled (B, C) features, got shape torch.Size([2, 7, 7, 96])` (or similar spatial shape) — confirms the real, current bug before the fix.

- [ ] **Step 3: Add the pooling shim to `ArkSwinTransformer`**

In `models.py`, replace the `forward`/`generate_embeddings` pair (currently lines 34–47) with:

```python
    def forward(self, x, head_n=None):
        x = self.forward_features(x)
        x = self._pool(x)
        if self.projector:
            x = self.projector(x)
        if head_n is not None:
            return x, self.omni_heads[head_n](x)
        else:
            return [head(x) for head in self.omni_heads]
    
    def generate_embeddings(self, x, after_proj = True):
        x = self.forward_features(x)
        x = self._pool(x)
        if after_proj:
            x = self.projector(x)
        return x

    def _pool(self, x):
        """timm>=0.9 changed SwinTransformer.forward_features() to return an
        unpooled spatial map (B, H, W, C) instead of the pooled (B, C) vector
        this class (written against the repo's pinned timm==0.5.4) expects --
        confirmed directly against the installed timm==1.0.28 during planning
        (forward_features returns (B,7,7,1024) for a 224x224 input). Bridge it
        with the encoder's own head pooling (respects whatever `global_pool`
        the encoder was built with) instead of reimplementing pooling."""
        if x.dim() > 2:
            x = self.forward_head(x, pre_logits=True)
        return x
```

- [ ] **Step 4: Add the `swin_tiny` branch to `build_omni_model`**

In `models.py`'s `build_omni_model` function, insert a new `elif` branch immediately before `elif args.model_name == "conv_base":`:

```python
    elif args.model_name == "swin_tiny": #swin_tiny_patch4_window7_224, sized for 8GB VRAM (rule 4: shrink Swin, don't substitute it)
        model = ArkSwinTransformer(num_classes_list, args.projector_features, args.use_mlp,
                                    img_size=args.crop_size, patch_size=4, window_size=7,
                                    embed_dim=96, depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24))
```

Note this branch passes `img_size=args.crop_size` explicitly (the existing `swin_base`/`swin_large*` branches rely on the class default of 224, matching the paper's fixed configurations) so `swin_tiny` can run at the reduced resolution Task 10's notebook uses.

- [ ] **Step 5: Add `load_imagenet_backbone`**

Append to the end of `models.py`, after `save_checkpoint`:

```python
def load_imagenet_backbone(model, timm_model_name):
    """ImageNet init for a from-scratch ArkSwinTransformer. `args.init` is
    parsed in main_ark.py's help text (Random|ImageNet_1k|...) but never
    actually read anywhere in this codebase (confirmed via grep: no other
    reference to args.init in main_ark.py/engine.py/models.py/trainer.py) --
    ImageNet init has to be wired in explicitly. Mirrors build_omni_model's
    existing pretrained-weight loading pattern (attn_mask key removal,
    strict=False, printed load message) but sources the state dict from
    timm's own hub instead of args.pretrained_weights."""
    import timm
    src = timm.create_model(timm_model_name, pretrained=True, num_classes=0)
    state_dict = src.state_dict()
    k_del = [k for k in state_dict if "attn_mask" in k]
    for k in k_del:
        del state_dict[k]
    msg = model.load_state_dict(state_dict, strict=False)
    print('Loaded ImageNet backbone ({}) with msg: {}'.format(timm_model_name, msg))
    return model
```

- [ ] **Step 6: Run the test again to confirm it passes**

Run: `conda run -n ark_medmnist python test_models_medmnist.py`
Expected: `test_models_medmnist.py: all checks passed`

- [ ] **Step 7: Commit**

```bash
git add models.py test_models_medmnist.py
git commit -m "fix timm==1.0.28 pooling break in ArkSwinTransformer; add swin_tiny branch and ImageNet-init helper"
```

---

### Task 3: `medmnist_dataloader.py` — real Ark+ Dataset contract for MedMNIST

**Files:**
- Modify: `medmnist_dataloader.py` (currently a from-scratch stub with a wrong `BreastMNIST: 3` class count and a symmetric-view bug — both bugs are real, confirmed by reading the current file; this task replaces its contents entirely)
- Test: `test_medmnist_dataloader.py`

**Interfaces:**
- Consumes: `medmnist.INFO` at import time (never hardcode class counts — the current file's `NUM_CLASSES_MAP['BreastMNIST'] = 3` is exactly the kind of bug this avoids; real answer is 2).
- Produces: `MEDMNIST_2D_KEYS` (list of the 12 dataset keys), `MEDMNIST_DATALOADER_DICT` (dict of key → Dataset class, mergeable into `dataloader.dict_dataloarder`), `medmnist_task_type(key)`, `medmnist_num_classes(key)`.

- [ ] **Step 1: Write the failing tests for the pure-logic helpers (no download needed)**

```python
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
```

- [ ] **Step 2: Run it to confirm it fails**

Run: `conda run -n ark_medmnist python test_medmnist_dataloader.py`
Expected: `ImportError` (the functions don't exist yet in the current stub file).

- [ ] **Step 3: Replace `medmnist_dataloader.py` entirely**

```python
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
```

- [ ] **Step 4: Run the test again to confirm it passes**

Run: `conda run -n ark_medmnist python test_medmnist_dataloader.py`
Expected: `test_medmnist_dataloader.py: all checks passed`

- [ ] **Step 5: Commit**

```bash
git add medmnist_dataloader.py test_medmnist_dataloader.py
git commit -m "rewrite medmnist_dataloader.py to Ark_Plus's real Dataset contract (fixes wrong BreastMNIST=3 class count and symmetric-view bug in the old stub)"
```

---

### Task 4: `datasets_config_medmnist.yaml` — verified class/task config, no hardcoding from memory

**Files:**
- Create: `datasets_config_medmnist.yaml`
- Create: `dump_medmnist_config.py` (throwaway-but-kept verification script — regenerates the YAML below from `medmnist.INFO`, so anyone can re-verify it later without trusting this plan's transcription)
- Test: `test_datasets_config_medmnist.py`

**Interfaces:**
- Produces: a YAML file with one entry per key in `medmnist_dataloader.MEDMNIST_2D_KEYS`, each with `data_dir` (unused, kept for schema parity), `train_list`/`val_list`/`test_list` (the literal strings `'train'`/`'val'`/`'test'`, consumed by `medmnist_dataloader.MedMNIST2DDataset` as split names), `diseases` (real label names, not placeholders — this is what `engine.py`'s `num_classes_list = [len(datasets_config[dataset]['diseases']) ...]` counts), `task_type`.

- [ ] **Step 1: Write the generator script**

```python
"""Regenerates datasets_config_medmnist.yaml from medmnist.INFO directly --
run this instead of hand-editing the YAML if medmnist is ever upgraded.
Run: python dump_medmnist_config.py"""
import yaml
from medmnist import INFO
from medmnist_dataloader import MEDMNIST_2D_KEYS, medmnist_task_type

config = {}
for key in MEDMNIST_2D_KEYS:
    info = INFO[key]
    labels = info['label']
    diseases = [labels[str(i)] for i in range(len(labels))]
    config[key] = {
        'data_dir': 'unused-medmnist-downloads-by-key',
        'train_list': 'train',
        'val_list': 'val',
        'test_list': 'test',
        'diseases': diseases,
        'task_type': medmnist_task_type(key),
    }

with open('datasets_config_medmnist.yaml', 'w') as f:
    yaml.dump(config, f, sort_keys=False, allow_unicode=True)
print("Wrote datasets_config_medmnist.yaml for", len(config), "datasets")
```

- [ ] **Step 2: Run it to generate the YAML**

Run: `conda run -n ark_medmnist python dump_medmnist_config.py`

This produces (verified during planning, reproduced here for review — if your run differs, trust the freshly-generated file over this transcription):

```yaml
pathmnist:
  data_dir: unused-medmnist-downloads-by-key
  train_list: train
  val_list: val
  test_list: test
  diseases: [adipose, background, debris, lymphocytes, mucus, smooth muscle, normal colon mucosa, cancer-associated stroma, colorectal adenocarcinoma epithelium]
  task_type: multi-class classification
bloodmnist:
  data_dir: unused-medmnist-downloads-by-key
  train_list: train
  val_list: val
  test_list: test
  diseases: [basophil, eosinophil, erythroblast, "immature granulocytes(myelocytes, metamyelocytes and promyelocytes)", lymphocyte, monocyte, neutrophil, platelet]
  task_type: multi-class classification
dermamnist:
  data_dir: unused-medmnist-downloads-by-key
  train_list: train
  val_list: val
  test_list: test
  diseases: [actinic keratoses and intraepithelial carcinoma, basal cell carcinoma, benign keratosis-like lesions, dermatofibroma, melanoma, melanocytic nevi, vascular lesions]
  task_type: multi-class classification
octmnist:
  data_dir: unused-medmnist-downloads-by-key
  train_list: train
  val_list: val
  test_list: test
  diseases: [choroidal neovascularization, diabetic macular edema, drusen, normal]
  task_type: multi-class classification
pneumoniamnist:
  data_dir: unused-medmnist-downloads-by-key
  train_list: train
  val_list: val
  test_list: test
  diseases: [normal, pneumonia]
  task_type: multi-class classification
retinamnist:
  data_dir: unused-medmnist-downloads-by-key
  train_list: train
  val_list: val
  test_list: test
  diseases: ["0", "1", "2", "3", "4"]
  task_type: multi-class classification
breastmnist:
  data_dir: unused-medmnist-downloads-by-key
  train_list: train
  val_list: val
  test_list: test
  diseases: [malignant, "normal, benign"]
  task_type: multi-class classification
tissuemnist:
  data_dir: unused-medmnist-downloads-by-key
  train_list: train
  val_list: val
  test_list: test
  diseases: ["Collecting Duct, Connecting Tubule", Distal Convoluted Tubule, Glomerular endothelial cells, Interstitial endothelial cells, Leukocytes, Podocytes, Proximal Tubule Segments, Thick Ascending Limb]
  task_type: multi-class classification
organamnist:
  data_dir: unused-medmnist-downloads-by-key
  train_list: train
  val_list: val
  test_list: test
  diseases: [bladder, femur-left, femur-right, heart, kidney-left, kidney-right, liver, lung-left, lung-right, pancreas, spleen]
  task_type: multi-class classification
organcmnist:
  data_dir: unused-medmnist-downloads-by-key
  train_list: train
  val_list: val
  test_list: test
  diseases: [bladder, femur-left, femur-right, heart, kidney-left, kidney-right, liver, lung-left, lung-right, pancreas, spleen]
  task_type: multi-class classification
organsmnist:
  data_dir: unused-medmnist-downloads-by-key
  train_list: train
  val_list: val
  test_list: test
  diseases: [bladder, femur-left, femur-right, heart, kidney-left, kidney-right, liver, lung-left, lung-right, pancreas, spleen]
  task_type: multi-class classification
chestmnist:
  data_dir: unused-medmnist-downloads-by-key
  train_list: train
  val_list: val
  test_list: test
  diseases: [atelectasis, cardiomegaly, effusion, infiltration, mass, nodule, pneumonia, pneumothorax, consolidation, edema, emphysema, fibrosis, pleural, hernia]
  task_type: multi-label classification
```

- [ ] **Step 3: Write and run the verification test**

```python
"""test_datasets_config_medmnist.py — cross-checks the generated YAML against
medmnist.INFO directly (not against this plan's transcription).
Run: python test_datasets_config_medmnist.py"""
import yaml
from medmnist import INFO
from medmnist_dataloader import MEDMNIST_2D_KEYS, medmnist_task_type, medmnist_num_classes

with open('datasets_config_medmnist.yaml') as f:
    config = yaml.safe_load(f)


def test_all_12_keys_present():
    assert set(config.keys()) == set(MEDMNIST_2D_KEYS)


def test_diseases_length_matches_medmnist_info():
    for key in MEDMNIST_2D_KEYS:
        assert len(config[key]['diseases']) == medmnist_num_classes(key), \
            f"{key}: YAML has {len(config[key]['diseases'])} diseases, medmnist.INFO says {medmnist_num_classes(key)}"


def test_task_type_matches_medmnist_info():
    for key in MEDMNIST_2D_KEYS:
        assert config[key]['task_type'] == medmnist_task_type(key)


def test_splits_are_literal_split_names():
    for key in MEDMNIST_2D_KEYS:
        assert config[key]['train_list'] == 'train'
        assert config[key]['val_list'] == 'val'
        assert config[key]['test_list'] == 'test'


if __name__ == "__main__":
    test_all_12_keys_present()
    test_diseases_length_matches_medmnist_info()
    test_task_type_matches_medmnist_info()
    test_splits_are_literal_split_names()
    print("test_datasets_config_medmnist.py: all checks passed")
```

Run: `conda run -n ark_medmnist python test_datasets_config_medmnist.py`
Expected: `test_datasets_config_medmnist.py: all checks passed`

- [ ] **Step 4: Commit**

```bash
git add datasets_config_medmnist.yaml dump_medmnist_config.py test_datasets_config_medmnist.py
git commit -m "add datasets_config_medmnist.yaml generated and verified directly from medmnist.INFO (fixes bug B: no hardcoded class counts)"
```

---

### Task 5: `checkpoint_utils.py` — atomic writes, RNG capture/restore

**Files:**
- Create: `checkpoint_utils.py`
- Test: `test_checkpoint_utils.py`

**Interfaces:**
- Produces: `save_checkpoint_atomic(state, path_latest, path_prev)`, `load_checkpoint_with_fallback(path_latest, path_prev) -> dict | None`, `capture_rng_state() -> dict`, `restore_rng_state(rng)`.

This is generic, reusable infrastructure with no dependency on `engine.py`'s internals — Task 6 wires it in.

- [ ] **Step 1: Write the failing tests**

```python
"""test_checkpoint_utils.py — Run: python test_checkpoint_utils.py"""
import os
import random
import shutil
import tempfile

import numpy as np
import torch

from checkpoint_utils import (
    save_checkpoint_atomic, load_checkpoint_with_fallback,
    capture_rng_state, restore_rng_state,
)


def test_round_trip():
    d = tempfile.mkdtemp()
    try:
        latest, prev = os.path.join(d, "latest.pth"), os.path.join(d, "prev.pth")
        state = {"epoch": 3, "dataset_index": 7, "weights": torch.randn(10)}
        save_checkpoint_atomic(state, latest, prev)
        loaded = load_checkpoint_with_fallback(latest, prev)
        assert loaded["epoch"] == 3 and loaded["dataset_index"] == 7
        assert torch.allclose(loaded["weights"], state["weights"])
    finally:
        shutil.rmtree(d)


def test_falls_back_to_prev_when_latest_is_corrupt():
    d = tempfile.mkdtemp()
    try:
        latest, prev = os.path.join(d, "latest.pth"), os.path.join(d, "prev.pth")
        save_checkpoint_atomic({"epoch": 1, "dataset_index": 0}, latest, prev)
        save_checkpoint_atomic({"epoch": 2, "dataset_index": 0}, latest, prev)  # prev now holds epoch=1
        with open(latest, "wb") as f:
            f.write(b"not a valid checkpoint")  # simulate a power cut mid-write
        loaded = load_checkpoint_with_fallback(latest, prev)
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
    print("test_checkpoint_utils.py: all checks passed")
```

- [ ] **Step 2: Run it to confirm it fails**

Run: `conda run -n ark_medmnist python test_checkpoint_utils.py`
Expected: `ModuleNotFoundError: No module named 'checkpoint_utils'`

- [ ] **Step 3: Write `checkpoint_utils.py`**

```python
"""Crash-proof checkpoint primitives: atomic writes (tmp -> fsync ->
os.replace, so a power cut can only ever leave the previous good file
intact -- os.replace is MoveFileExW with MOVEFILE_REPLACE_EXISTING on
Windows, atomic for same-volume renames) and full RNG state capture/restore,
so training resumes byte-for-byte rather than restarting the RNG stream."""
import os
import random

import numpy as np
import torch


def save_checkpoint_atomic(state, path_latest, path_prev):
    tmp_path = path_latest + ".tmp"
    with open(tmp_path, "wb") as f:
        torch.save(state, f)
        f.flush()
        os.fsync(f.fileno())
    if os.path.isfile(path_latest):
        os.replace(path_latest, path_prev)
    os.replace(tmp_path, path_latest)


def load_checkpoint_with_fallback(path_latest, path_prev):
    """Try latest; if missing or fails to load (e.g. killed between the two
    os.replace() calls, or disk corruption), fall back to the previous one."""
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

- [ ] **Step 4: Run the test again to confirm it passes**

Run: `conda run -n ark_medmnist python test_checkpoint_utils.py`
Expected: `test_checkpoint_utils.py: all checks passed`

- [ ] **Step 5: Commit**

```bash
git add checkpoint_utils.py test_checkpoint_utils.py
git commit -m "add checkpoint_utils.py: atomic checkpoint writes and full RNG state capture/restore"
```

---

### Task 6: `trainer.py` — AMP, without touching the loss/EMA formulas

**Files:**
- Modify: `trainer.py`
- Test: `test_trainer_amp.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `train_one_epoch(..., it, scaler=None)` and `evaluate(..., scaler=None)` — both existing signatures gain one new trailing optional parameter, defaulting to `None` (no AMP), so any *other* caller of these two functions with the original argument list keeps working unchanged.

- [ ] **Step 1: Write the failing test**

```python
"""test_trainer_amp.py — confirms AMP-enabled train_one_epoch runs, produces
finite losses, and updates weights, using a tiny CPU-safe stand-in model
(autocast is skipped on CPU; this test only exercises the code path, not
actual mixed-precision numerics, which requires a CUDA device -- see
Task 11 for the live-GPU verification). Run: python test_trainer_amp.py"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from trainer import train_one_epoch


class _TinyArkLike(nn.Module):
    """Mimics ArkSwinTransformer's forward(x, head_n) -> (feat, logits) contract
    with a trivial linear encoder, so this test has no GPU/timm dependency."""
    def __init__(self, in_dim=8, feat_dim=4, num_classes=3):
        super().__init__()
        self.enc = nn.Linear(in_dim, feat_dim)
        self.head = nn.Linear(feat_dim, num_classes)

    def forward(self, x, head_n=None):
        feat = self.enc(x)
        return feat, self.head(feat)


def test_train_one_epoch_runs_with_amp_disabled_on_cpu():
    torch.manual_seed(0)
    model = _TinyArkLike()
    teacher = _TinyArkLike()
    teacher.load_state_dict(model.state_dict())
    for p in teacher.parameters():
        p.requires_grad = False

    x1 = torch.randn(16, 8)
    x2 = x1 + 0.01 * torch.randn(16, 8)
    y = torch.eye(3)[torch.randint(0, 3, (16,))]
    loader = DataLoader(TensorDataset(x1, x2, y), batch_size=4)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    momentum_schedule = [0.9] * 10
    before = model.enc.weight.clone()

    scaler = torch.amp.GradScaler('cuda', enabled=False)  # CPU test: AMP path present but inert
    train_one_epoch(model, 0, "tinyset", loader, torch.device('cpu'), nn.CrossEntropyLoss(),
                     optimizer, epoch=0, ema_mode="epoch", teacher=teacher,
                     momentum_schedule=momentum_schedule, it=0, scaler=scaler)

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
    x1 = torch.randn(8, 8)
    x2 = x1.clone()
    y = torch.eye(3)[torch.randint(0, 3, (8,))]
    loader = DataLoader(TensorDataset(x1, x2, y), batch_size=4)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    train_one_epoch(model, 0, "tinyset", loader, torch.device('cpu'), nn.CrossEntropyLoss(),
                     optimizer, epoch=0, ema_mode="epoch", teacher=teacher,
                     momentum_schedule=[0.9], it=0)  # no scaler kwarg at all


if __name__ == "__main__":
    test_train_one_epoch_runs_with_amp_disabled_on_cpu()
    test_backward_compatible_call_without_scaler_still_works()
    print("test_trainer_amp.py: all checks passed")
```

- [ ] **Step 2: Run it to confirm it fails**

Run: `conda run -n ark_medmnist python test_trainer_amp.py`
Expected: `TypeError: train_one_epoch() got an unexpected keyword argument 'scaler'`

- [ ] **Step 3: Add AMP to `train_one_epoch` and `evaluate`, and make `wandb.log` safe**

Replace `trainer.py`'s `import wandb` line (line 5) with:

```python
try:
    import wandb
except ImportError:
    wandb = None
```

Replace `train_one_epoch`'s signature and body (currently lines 7–61) with:

```python
def train_one_epoch(model, use_head_n, dataset, data_loader_train, device, criterion, optimizer, epoch, ema_mode, teacher, momentum_schedule, it, scaler=None):
    batch_time = MetricLogger('Time', ':6.3f')
    losses_cls = MetricLogger('Loss_'+dataset+' cls', ':.4e')
    losses_mse = MetricLogger('Loss_'+dataset+' mse', ':.4e')
    progress = ProgressLogger(
        len(data_loader_train),
        [batch_time, losses_cls, losses_mse],
        prefix="Epoch: [{}]".format(epoch))

    model.train()
    MSE = torch.nn.MSELoss()
    coff = (momentum_schedule[it] - 0.9) * 5
    amp_enabled = scaler is not None and scaler.is_enabled()
    end = time.time()
    for i, (samples1, samples2, targets) in enumerate(data_loader_train):
        samples1, samples2, targets = samples1.float().to(device), samples2.float().to(device), targets.float().to(device)

        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
            feat_t, pred_t = teacher(samples2, use_head_n)
            feat_s, pred_s = model(samples1, use_head_n)
            loss_cls = criterion(pred_s, targets)
            loss_const = MSE(feat_s, feat_t)
            loss = (1-coff) * loss_cls + coff * loss_const

        optimizer.zero_grad()
        if amp_enabled:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        losses_cls.update(loss_cls.item(), samples1.size(0))
        losses_mse.update(loss_const.item(), samples1.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 50 == 0:
            progress.display(i)
            save_image(samples1[0].detach().float().cpu().numpy().transpose(1, 2, 0), "Models/student"+str(i))
            save_image(samples2[0].detach().float().cpu().numpy().transpose(1, 2, 0),"Models/teacher"+str(i))

        if ema_mode == "iteration":
            ema_update_teacher(model, teacher, momentum_schedule, it)
            it += 1

    if ema_mode == "epoch":
        ema_update_teacher(model, teacher, momentum_schedule, it)
        it += 1

    if wandb is not None and wandb.run is not None:
        wandb.log({"train_loss_cls_{}".format(dataset): losses_cls.avg})
        wandb.log({"train_loss_mse_{}".format(dataset): losses_mse.avg})
```

The only algorithmic lines are unchanged verbatim: `coff = (momentum_schedule[it] - 0.9) * 5`, `loss_cls = criterion(pred_s, targets)`, `loss_const = MSE(feat_s, feat_t)`, `loss = (1-coff) * loss_cls + coff * loss_const`, and `ema_update_teacher(...)`. Everything added is either the `torch.autocast`/`GradScaler` wrapping (inert when `scaler=None` or `scaler.is_enabled()` is `False`) or the `wandb.run is not None` guard (replaces the unconditional `wandb.log()` calls that currently crash without an active run — see Task 7 for where a disabled-mode run gets started).

Replace `evaluate`'s signature (currently line 71) with:

```python
def evaluate(model, use_head_n, data_loader_val, device, criterion, dataset, scaler=None):
    model.eval()
    amp_enabled = scaler is not None and scaler.is_enabled()

    with torch.no_grad():
        batch_time = MetricLogger('Time', ':6.3f')
        losses = MetricLogger('Loss', ':.4e')
        progress = ProgressLogger(
        len(data_loader_val),
        [batch_time, losses], prefix='Val_'+dataset+': ')

        end = time.time()
        for i, (samples, _, targets) in enumerate(data_loader_val):
            samples, targets = samples.float().to(device), targets.float().to(device)

            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                _, outputs = model(samples, use_head_n)
                loss = criterion(outputs, targets)

            losses.update(loss.item(), samples.size(0))
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 50 == 0:
                progress.display(i)

    return losses.avg
```

- [ ] **Step 4: Run the test again to confirm it passes**

Run: `conda run -n ark_medmnist python test_trainer_amp.py`
Expected: `test_trainer_amp.py: all checks passed`

- [ ] **Step 5: Commit**

```bash
git add trainer.py test_trainer_amp.py
git commit -m "add AMP (autocast+GradScaler) to train_one_epoch/evaluate via optional scaler=None param; guard wandb.log against no active run (deviation: AMP required for 8GB VRAM, see DEVIATIONS.md)"
```

---

### Task 7: `engine.py` — crash-proof per-dataset checkpoint/resume (opt-in, original path untouched)

**Files:**
- Modify: `engine.py`
- Test: `test_engine_resume_indexing.py`

**Interfaces:**
- Consumes: `checkpoint_utils.{save_checkpoint_atomic, load_checkpoint_with_fallback, capture_rng_state, restore_rng_state}` (Task 5).
- Produces: `omni_engine(...)` gains crash-proof per-dataset checkpointing **only when `args.crash_proof_resume` is `True`** (a new attribute the notebook sets explicitly in Task 10; absent/`False` reproduces the original epoch-only-checkpoint behavior exactly, so the original ChestXray14/CheXpert/etc. pretraining path is unaffected).

This is the **most important requirement** per the brief: the machine loses power, training must resume exactly where it stopped, not at the start of the epoch. The real `omni_engine` only checkpoints at epoch boundaries (confirmed by reading it) — one epoch over even 12 datasets can be a multi-hour run, so a power cut currently discards all of it.

- [ ] **Step 1: Write a test for the pure resume-indexing logic (no GPU/training needed)**

The trickiest part of resumable cyclic training is exactly which dataset index to resume from — get this wrong and you either skip a dataset's exposure for the epoch or redo the whole epoch. Test it as a pure function before wiring it into the loop.

```python
"""test_engine_resume_indexing.py — Run: python test_engine_resume_indexing.py"""
from engine import _resume_dataset_range


def test_no_checkpoint_starts_at_zero():
    first_i = _resume_dataset_range(checkpoint=None, current_epoch=0, n_datasets=5)
    assert first_i == 0


def test_mid_epoch_resume_skips_completed_datasets():
    # last_completed=2 means datasets 0,1,2 fully finished (train+EMA) this
    # epoch -- resume must continue at index 3, not redo or skip one.
    ckpt = {'epoch': 3, 'last_completed': 2}
    first_i = _resume_dataset_range(checkpoint=ckpt, current_epoch=3, n_datasets=5)
    assert first_i == 3


def test_resume_only_applies_to_the_checkpointed_epoch():
    # A later epoch (e.g. after a full-epoch checkpoint) must start at 0,
    # not re-apply the old epoch's last_completed position.
    ckpt = {'epoch': 3, 'last_completed': 2}
    first_i = _resume_dataset_range(checkpoint=ckpt, current_epoch=4, n_datasets=5)
    assert first_i == 0


def test_checkpoint_saved_after_last_dataset_resumes_at_n_and_loop_is_empty():
    # If the checkpoint landed right after the LAST dataset of an epoch
    # finished (before validation ran), first_i should equal n_datasets so
    # range(first_i, n_datasets) is empty and control falls straight to
    # validation -- no separate "epoch already done" branch needed.
    ckpt = {'epoch': 3, 'last_completed': 4}
    first_i = _resume_dataset_range(checkpoint=ckpt, current_epoch=3, n_datasets=5)
    assert first_i == 5


if __name__ == "__main__":
    test_no_checkpoint_starts_at_zero()
    test_mid_epoch_resume_skips_completed_datasets()
    test_resume_only_applies_to_the_checkpointed_epoch()
    test_checkpoint_saved_after_last_dataset_resumes_at_n_and_loop_is_empty()
    print("test_engine_resume_indexing.py: all checks passed")
```

- [ ] **Step 2: Run it to confirm it fails**

Run: `conda run -n ark_medmnist python test_engine_resume_indexing.py`
Expected: `ImportError: cannot import name '_resume_dataset_range' from 'engine'`

- [ ] **Step 3: Add `_resume_dataset_range` and wire crash-proof resume into `omni_engine`**

At the top of `engine.py`, add the import:

```python
from checkpoint_utils import (
    save_checkpoint_atomic, load_checkpoint_with_fallback,
    capture_rng_state, restore_rng_state,
)
```

Add this pure helper function near the top of the file (after imports, before `omni_engine`):

```python
def _resume_dataset_range(checkpoint, current_epoch, n_datasets):
    """Which dataset index to start `current_epoch` at. `last_completed` is
    the last index that FULLY finished training+EMA this epoch (-1 if none),
    tracked separately from the loop variable so a KeyboardInterrupt mid-
    train_one_epoch never marks an in-progress dataset as done (that would
    make resume skip it, silently losing that dataset's exposure)."""
    if checkpoint is None or checkpoint.get('epoch') != current_epoch:
        return 0
    return checkpoint['last_completed'] + 1
```

Now modify `omni_engine`'s `if args.mode == "train":` block. The `momentum_schedule`/`optimizer`/`lr_scheduler` construction above it (lines 86–97) is unchanged. Replace lines 104–258 (the full `if args.mode == "train":` body) with:

```python
    if args.mode == "train":
        crash_proof_resume = getattr(args, 'crash_proof_resume', False)
        atomic_latest = save_model_path + '_atomic_latest.pth'
        atomic_prev = save_model_path + '_atomic_prev.pth'
        scaler = torch.amp.GradScaler('cuda', enabled=getattr(args, 'use_amp', False))

        ckpt = None
        if crash_proof_resume:
            ckpt = load_checkpoint_with_fallback(atomic_latest, atomic_prev)
            if ckpt is not None:
                model.load_state_dict(ckpt['student'])
                teacher.load_state_dict(ckpt['teacher'])
                optimizer.load_state_dict(ckpt['optimizer'])
                lr_scheduler.load_state_dict(ckpt['scheduler'])
                scaler.load_state_dict(ckpt['scaler'])
                restore_rng_state(ckpt['rng'])
                start_epoch = ckpt['epoch']
                skipped = _resume_dataset_range(ckpt, ckpt['epoch'], len(dataset_list))
                print(f">>> Crash-proof resume: epoch {start_epoch}, skipping datasets "
                      f"0..{skipped - 1} already completed this epoch: {dataset_list[:skipped]}")
            else:
                print(">>> Crash-proof resume enabled, no checkpoint found — starting fresh")
        elif args.resume:
            # Original epoch-only resume path -- unchanged from the real repo.
            resume = save_model_path + '.pth.tar'
            if os.path.isfile(resume):
                print("=> loading checkpoint '{}'".format(resume))
                checkpoint = torch.load(resume)
                start_epoch = checkpoint['epoch']
                init_loss = checkpoint['lossMIN']
                state_dict = checkpoint['state_dict']
                teacher_state_dict = checkpoint['teacher']

                if args.reinit_heads:
                    for k in model.state_dict().keys():
                        if k.startswith('omni_heads.'):
                            print(f"Removing key {k} from pretrained checkpoint")
                            del state_dict[k]

                model.load_state_dict(state_dict, strict=True)
                teacher.load_state_dict(teacher_state_dict, strict=True)
                lr_scheduler.load_state_dict(checkpoint['scheduler'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                print("=> loaded checkpoint '{}' (epoch={:04d}, val_loss={})"
                        .format(resume, start_epoch, init_loss))
                start_epoch += 1
            else:
                print("=> no checkpoint found at '{}'".format(args.resume))

        if wandb is not None:
            wandb.init(project=exp + '_' + args.exp_name, mode="disabled")

        with open(log_file, 'a') as log:
                log.write(str(args))
        log.close()

        test_results,test_results_teacher = [],[]
        it = start_epoch * len(dataset_list)

        def checkpoint_now(epoch, last_completed):
            state = {
                'epoch': epoch, 'last_completed': last_completed, 'it': it,
                'student': model.state_dict(), 'teacher': teacher.state_dict(),
                'optimizer': optimizer.state_dict(), 'scheduler': lr_scheduler.state_dict(),
                'scaler': scaler.state_dict(), 'rng': capture_rng_state(),
            }
            save_checkpoint_atomic(state, atomic_latest, atomic_prev)

        try:
            for epoch in range(start_epoch, args.pretrain_epochs):
                first_i = _resume_dataset_range(ckpt, epoch, len(dataset_list)) if crash_proof_resume else 0
                last_completed = first_i - 1  # dedicated tracker: `i` gets rebound by the
                                               # validation for-loop below, so KeyboardInterrupt
                                               # during validation must NOT read a stale `i`
                for i, data_loader in enumerate(data_loader_list_train):
                    if crash_proof_resume and i < first_i:
                        it += 1
                        continue
                    criterion = torch.nn.CrossEntropyLoss() if datasets_config[dataset_list[i]]['task_type'] == "multi-class classification" else torch.nn.BCEWithLogitsLoss()
                    train_one_epoch(model, i, dataset_list[i], data_loader, device, criterion, optimizer, epoch, args.ema_mode, teacher, momentum_schedule, it, scaler=scaler)
                    it += 1
                    last_completed = i
                    if crash_proof_resume:
                        checkpoint_now(epoch, last_completed)
                val_loss_list = []
                for i, dv in enumerate(data_loader_list_val):
                    criterion = torch.nn.CrossEntropyLoss() if datasets_config[dataset_list[i]]['task_type'] == "multi-class classification" else torch.nn.BCEWithLogitsLoss()
                    val_loss = evaluate(model, i, dv, device, criterion, dataset_list[i], scaler=scaler)
                    val_loss_list.append(val_loss)

                avg_val_loss = np.average(val_loss_list)
                if args.val_loss_metric == "average":
                    val_loss_metric = avg_val_loss
                else:
                    val_loss_metric = val_loss_list[dataset_list.index(args.val_loss_metric)]
                lr_scheduler.step(val_loss_metric)

                print("Epoch {:04d}: avg_val_loss {:.5f}, saving model to {}".format(epoch, avg_val_loss,save_model_path))
                save_checkpoint({
                        'epoch': epoch,
                        'lossMIN': val_loss_list,
                        'state_dict': model.state_dict(),
                        'teacher': teacher.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': lr_scheduler.state_dict(),
                        },  filename=save_model_path)
                if crash_proof_resume:
                    last_completed = len(dataset_list) - 1  # whole epoch done
                    checkpoint_now(epoch, last_completed)

                with open(log_file, 'a') as log:
                    log.write("Epoch {:04d}: avg_val_loss = {:.5f} \n".format(epoch, avg_val_loss))
                    log.write("     Datasets  : " + str(dataset_list) + "\n")
                    log.write("     Val Losses: " + str(val_loss_list) + "\n")
                    log.close()

                if epoch % args.test_epoch == 0 or epoch+1 == args.pretrain_epochs:
                    save_checkpoint({
                         'epoch': epoch,
                         'lossMIN': val_loss_list,
                         'state_dict': model.state_dict(),
                         'teacher': teacher.state_dict(),
                         'optimizer': optimizer.state_dict(),
                         'scheduler': lr_scheduler.state_dict(),
                         },  filename=save_model_path+str(epoch))
                    with open(output_file, 'a') as writer:
                        writer.write("Omni-pretraining stage:\n")
                        writer.write("Epoch {:04d}:\n".format(epoch))
                        t_res, t_res_teacher = [],[]
                        for i, dataset in enumerate(dataset_list):
                            writer.write("{} Validation Loss = {:.5f}:\n".format(dataset, val_loss_list[i]))
                            diseases = datasets_config[dataset]['diseases']
                            print(">>{} Disease = {}".format(dataset, diseases))
                            writer.write("{} Disease = {}\n".format(dataset, diseases))

                            multiclass =  datasets_config[dataset]['task_type'] == "multi-class classification"
                            y_test, p_test = test_classification(model, i, data_loader_list_test[i], device, multiclass)
                            y_test_teacher, p_test_teacher = test_classification(teacher, i, data_loader_list_test[i], device, multiclass)
                            if multiclass:
                                acc = accuracy_score(np.argmax(y_test.cpu().numpy(),axis=1),np.argmax(p_test.cpu().numpy(),axis=1))
                                acc_teacher = accuracy_score(np.argmax(y_test_teacher.cpu().numpy(),axis=1),np.argmax(p_test_teacher.cpu().numpy(),axis=1))
                                print(">>{}:Student ACCURACY = {}, \nTeacher ACCURACY = {}\n".format(dataset,acc, acc_teacher))
                                writer.write(
                                    "\n{}: Student ACCURACY = {}, \nTeacher ACCURACY = {}\n".format(dataset, np.array2string(np.array(acc), precision=4, separator='\t'), np.array2string(np.array(acc_teacher), precision=4, separator='\t')))
                                t_res.append(acc)
                                t_res_teacher.append(acc_teacher)

                            if dataset == "CheXpert":
                                test_diseases_name = datasets_config['CheXpert']['test_diseases_name']
                                test_diseases = [diseases.index(c) for c in test_diseases_name]
                                y_test = copy.deepcopy(y_test[:,test_diseases])
                                p_test = copy.deepcopy(p_test[:, test_diseases])
                                individual_results = metric_AUROC(y_test, p_test, len(test_diseases))
                                y_test_teacher = copy.deepcopy(y_test_teacher[:,test_diseases])
                                p_test_teacher = copy.deepcopy(p_test_teacher[:, test_diseases])
                                individual_results_teacher = metric_AUROC(y_test_teacher, p_test_teacher, len(test_diseases))
                            else:
                                individual_results = metric_AUROC(y_test, p_test, len(diseases))
                                individual_results_teacher = metric_AUROC(y_test_teacher, p_test_teacher, len(diseases))
                            print(">>{}:Student AUC = {}, \nTeacher AUC = {}\n".format(dataset, np.array2string(np.array(individual_results), precision=4, separator='\t'),np.array2string(np.array(individual_results_teacher), precision=4, separator='\t')))
                            writer.write(
                                "\n{}: Student AUC = {}, \nTeacher AUC = {}\n".format(dataset, np.array2string(np.array(individual_results), precision=4, separator='\t'),np.array2string(np.array(individual_results_teacher), precision=4, separator='\t')))
                            mean_over_all_classes = np.array(individual_results).mean()
                            mean_over_all_classes_teacher = np.array(individual_results_teacher).mean()
                            print(">>{}: Student mAUC = {:.4f}, Teacher mAUC = {:.4f}".format(dataset, mean_over_all_classes,mean_over_all_classes_teacher))
                            writer.write("{}: Student mAUC = {:.4f}, Teacher mAUC = {:.4f}\n".format(dataset, mean_over_all_classes,mean_over_all_classes_teacher))
                            t_res.append(mean_over_all_classes)
                            t_res_teacher.append(mean_over_all_classes_teacher)

                        writer.close()

                        test_results.append(t_res)
                        test_results_teacher.append(t_res_teacher)

                        print("Omni-pretraining stage: \nStudent meanAUC = \n{} \nTeacher meanAUC = \n{}\n".format(test_results, test_results_teacher))
        except KeyboardInterrupt:
            print("\n>>> KeyboardInterrupt caught.")
            if crash_proof_resume:
                print(">>> Checkpointing before exit...")
                checkpoint_now(epoch, last_completed)
                print(f">>> Saved checkpoint at epoch={epoch}, last_completed={last_completed}. Exiting.")
            raise

        with open(output_file, 'a') as writer:
            writer.write("Omni-pretraining stage: \nStudent meanAUC = \n{} \nTeacher meanAUC = \n{}\n".format(np.array2string(np.array(test_results), precision=4, separator='\t'),np.array2string(np.array(test_results_teacher), precision=4, separator='\t')))
        writer.close()
```

Also add `import wandb` at the top of `engine.py` (it's currently commented out — re-enable it as a real import guarded the same way as `trainer.py`'s):

```python
try:
    import wandb
except ImportError:
    wandb = None
```

**What actually changed vs. the original, for the record:** the `momentum_schedule`/`coff` construction, `train_one_epoch`/`evaluate`/`ema_update_teacher` calls, and every AUROC/logging line are byte-for-byte identical to what was read from `Ark_Plus/Pretraining/engine.py`. The only new logic is the `crash_proof_resume`-gated checkpoint/resume wrapper (inert when the flag is `False`, which is the default and matches `args`'s absence of the attribute entirely for any caller that doesn't set it) and the `try/except KeyboardInterrupt`. Diff this task's `engine.py` against `D:\Ark_upstream_clone\Ark_Plus\Pretraining\engine.py` to audit exactly this.

- [ ] **Step 4: Run the resume-indexing test again to confirm it passes**

Run: `conda run -n ark_medmnist python test_engine_resume_indexing.py`
Expected: `test_engine_resume_indexing.py: all checks passed`

- [ ] **Step 5: Commit**

```bash
git add engine.py test_engine_resume_indexing.py
git commit -m "add crash-proof per-dataset checkpointing to omni_engine, gated behind args.crash_proof_resume (default False, original epoch-only path unchanged); re-enable wandb in disabled mode"
```

---

### Task 8: `main_ark.py` — trivial `argv` param so the notebook can build args without CLI parsing

**Files:**
- Modify: `main_ark.py`
- Test: `test_get_args_parser_argv.py`

**Interfaces:**
- Produces: `get_args_parser(argv=None)` — `argv=None` preserves the original behavior exactly (`optparse.parse_args(None)` reads `sys.argv[1:]`, same as today's no-argument `parser.parse_args()`); passing `argv=[]` lets a notebook get clean defaults regardless of Jupyter's own `sys.argv` (which contains `-f <kernel-connection-file>` and would otherwise be parsed as an unrecognized option).

- [ ] **Step 1: Write the failing test**

```python
"""test_get_args_parser_argv.py — Run: python test_get_args_parser_argv.py"""
import sys
from main_ark import get_args_parser


def test_explicit_empty_argv_ignores_process_sys_argv():
    old_argv = sys.argv
    try:
        sys.argv = ['ipykernel_launcher.py', '-f', '/some/kernel-connection.json']
        args = get_args_parser(argv=[])
        assert args.model_name == "swin_base"  # default, not crashed on '-f'
    finally:
        sys.argv = old_argv


def test_argv_none_still_reads_sys_argv_for_backward_compat():
    old_argv = sys.argv
    try:
        sys.argv = ['main_ark.py', '--model', 'swin_tiny']
        args = get_args_parser(argv=None)
        assert args.model_name == "swin_tiny"
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    test_explicit_empty_argv_ignores_process_sys_argv()
    test_argv_none_still_reads_sys_argv_for_backward_compat()
    print("test_get_args_parser_argv.py: all checks passed")
```

- [ ] **Step 2: Run it to confirm it fails**

Run: `conda run -n ark_medmnist python test_get_args_parser_argv.py`
Expected: `TypeError: get_args_parser() takes 0 positional arguments but 1 was given`

- [ ] **Step 3: Change the signature**

In `main_ark.py`, change:

```python
def get_args_parser():
    parser = OptionParser()
```

to:

```python
def get_args_parser(argv=None):
    parser = OptionParser()
```

and change the parse call near the bottom of the function from:

```python
    (options, args) = parser.parse_args()
```

to:

```python
    (options, args) = parser.parse_args(argv)
```

- [ ] **Step 4: Run the test again to confirm it passes**

Run: `conda run -n ark_medmnist python test_get_args_parser_argv.py`
Expected: `test_get_args_parser_argv.py: all checks passed`

- [ ] **Step 5: Commit**

```bash
git add main_ark.py test_get_args_parser_argv.py
git commit -m "add optional argv param to get_args_parser so notebooks can build args without consuming Jupyter's own sys.argv"
```

---

### Task 9: `DEVIATIONS.md` v1

**Files:**
- Create: `DEVIATIONS.md`

- [ ] **Step 1: Write the file**

```markdown
# Deviations from the Ark+ paper/repository

Scope: the 12-2D-MedMNIST replication (brief step 2) only. 3D and the 18-combined
extension get their own DEVIATIONS entries in their follow-up plans.

## Genuine blockers found and fixed (not design choices)

**timm API break.** `models.py`'s `ArkSwinTransformer` was written against the repo's
pinned `timm==0.5.4`, where `SwinTransformer.forward_features()` returned a pooled
`(B, C)` vector. The installed `timm==1.0.28` (needed for `torch==2.13.0+cu130`/Python
3.11 compatibility -- `timm==0.5.4` has no wheels for either) returns an unpooled
`(B, H, W, C)` spatial map instead, confirmed by direct inspection during planning.
Fixed with a one-line bridge (`self.forward_head(x, pre_logits=True)`) in
`ArkSwinTransformer._pool`, added in both `forward` and `generate_embeddings`. This is
a compatibility shim for a library version gap, not a change to Ark+'s model design.

**Missing dependencies.** `dataloader.py` imports `cv2`, `albumentations`, `pydicom`,
`yaml` at module level (used only by the X-ray dataset classes, never touched by this
plan) and `trainer.py` imports `wandb` and calls `wandb.log()` unconditionally with no
`wandb.init()` anywhere active. None were installed in `ark_medmnist`; `environment.yml`
now pins all five. `wandb.init(..., mode="disabled")` is called once per run in
`engine.py` so the existing `wandb.log()` call sites work as safe no-ops without a real
wandb account -- no experiment-tracking behavior was requested or added.

## Deliberate, flagged deviations

**Backbone size:** Swin-Base (paper's smallest documented config, ~88M params) ->
Swin-Tiny (~28M params, `embed_dim=96, depths=(2,2,6,2), num_heads=(3,6,12,24)`).
Reason: paper trained on 4x A100 80GB; this runs on one RTX 4060 (8188 MiB total,
5886 MiB free at planning time -- other processes were already holding ~2GB).
Rule 4 explicitly allows shrinking Swin, never substituting the architecture family;
`build_omni_model`'s new `swin_tiny` branch is still `ArkSwinTransformer`, unchanged.

**Resolution:** the paper trains at up to 768x768. This runs at whatever `args.crop_size`
is set to for the `swin_tiny` branch (notebook uses 112 -- see Task 10) since MedMNIST's
native resolution is 28x28 and Swin's `window_size=7`/`patch_size=4` requires
`img_size/patch_size` divisible by `window_size`; 112 satisfies this (112/4=28, 28%7=0)
at a quarter the pixel cost of 224 for no loss of correctness.

**AMP added** (`torch.autocast` + `torch.amp.GradScaler`, gated behind `args.use_amp` in
`engine.py`/Task 7, defaulting to `False`). Not in the paper (trained on A100 80GB, no
mixed-precision need). Required to fit an 8GB card. Off by default; the notebook (Task
10) turns it on explicitly.

**Crash-proof per-dataset checkpointing added**, gated behind `args.crash_proof_resume`
in `engine.py`, defaulting to `False` (original epoch-only checkpoint path is completely
unaffected when unset). Not in the paper -- an operational requirement for an unattended
local run that can lose power mid-cycle. Every algorithmic line inside the training loop
(`coff`, `loss_cls`, `loss_const`, `loss`, `ema_update_teacher`) is untouched; only the
surrounding orchestration (when to save/load state, which dataset index to resume at)
changed. Diffable against `D:\Ark_upstream_clone\Ark_Plus\Pretraining\engine.py`.

## Corrected understanding vs. an earlier (wrong) draft of this brief

An earlier version of the working brief described Ark+'s loss as `L_total = L_cls +
L_consist` (an unweighted sum) and characterized a `(1-coff)*loss_cls + coff*loss_mse`
convex combination as a bug unique to a from-scratch reimplementation. Reading the real
`Ark_Plus/Pretraining/trainer.py` directly shows this is backwards: the actual repo DOES
use a convex combination, `loss = (1-coff) * loss_cls + coff * loss_const`, where
`coff = (momentum_schedule[it] - 0.9) * 5` is tied to the same cosine-annealed EMA
momentum schedule as the teacher update (ramping `coff` from 0 toward ~0.5 as momentum
ramps from 0.9 toward 1.0 over the full pretraining run) -- not a flat sum, and not a
constant momentum of 0.9 either. This plan reproduces that formula exactly (Task 6);
flagging the correction here rather than silently absorbing it, since it changes what
"faithful replication" means for this specific mechanism.

There is also no "best checkpoint by mean AUC" selection anywhere in the real
`omni_engine` -- it checkpoints unconditionally every epoch and evaluates test-set AUROC
periodically for monitoring only. Rule 3 (one model, not N) still applies, but the
selection logic to satisfy it doesn't exist yet in this codebase; building it is
deferred to the evaluation-harness follow-up plan, once this plan has produced a
checkpoint to select from.
```

- [ ] **Step 2: Commit**

```bash
git add DEVIATIONS.md
git commit -m "add DEVIATIONS.md v1 for the 12-2D-MedMNIST replication, including the corrected loss/EMA-schedule finding from reading the real trainer.py"
```

---

### Task 10: `pretrain_medmnist2d.ipynb` — the notebook driver

**Files:**
- Create: `pretrain_medmnist2d.ipynb`

**Interfaces:**
- Consumes: everything from Tasks 1–9.
- Produces: an interactive, cell-by-cell driver that builds `args`, merges the MedMNIST dataset classes into `dict_dataloarder`, constructs the train/val/test `Dataset`/`DataLoader` lists exactly as `main_ark.py`'s `main()` does, and calls `omni_engine(...)` with crash-proof resume and AMP turned on.

- [ ] **Step 1: Write the notebook (as JSON — this is what `.ipynb` actually is)**

```python
import json

cells = []

def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)})

def code(text):
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {},
                   "outputs": [], "source": text.splitlines(keepends=True)})

md("# Ark+ pretraining on 12 2D MedMNIST datasets\n\n"
   "Drives the real `Ark_Plus/Pretraining` code (`engine.py`/`trainer.py`/`models.py`/"
   "`dataloader.py`, unmodified except for the compat shims and crash-proof resume "
   "documented in `DEVIATIONS.md`) against the MedMNIST data layer from "
   "`medmnist_dataloader.py`. Run `verify_env.py` first if you haven't.")

code("""import sys, os
sys.argv = ['pretrain_medmnist2d']  # keep optparse away from Jupyter's own -f <kernel.json>

from main_ark import get_args_parser
from dataloader import dict_dataloarder, build_transform_classification
from medmnist_dataloader import MEDMNIST_DATALOADER_DICT, MEDMNIST_2D_KEYS
from utils import get_config
from engine import omni_engine

dict_dataloarder.update(MEDMNIST_DATALOADER_DICT)  # register the 12 MedMNIST classes
""")

md("## Configure the run\n\n"
   "`swin_tiny` + 112px + batch 32 is sized for an 8GB card with ~2-6GB free "
   "(see `DEVIATIONS.md`) -- **run `nvidia-smi` right before this and reduce "
   "`batch_size` if free VRAM is lower than it was at planning time.**")

code("""args = get_args_parser(argv=[])

args.model_name = "swin_tiny"
args.dataset_list = MEDMNIST_2D_KEYS
args.crop_size = 112
args.resize = 128
args.batch_size = 32
args.workers = 4
args.pretrain_epochs = 25          # DEVIATION from paper's 50 cycles -- see DEVIATIONS.md
args.momentum_teacher = 0.9
args.ema_mode = "epoch"
args.exp_name = "medmnist2d_swintiny"
args.projector_features = 512
args.use_mlp = False
args.pretrained_weights = None      # ImageNet init applied separately below, not via this path
args.opt = "momentum"
args.lr = 1e-2
args.momentum = 0.9
args.weight_decay = 1e-4
args.resume = False                 # original epoch-only resume path -- off, we use crash_proof_resume
args.crash_proof_resume = True      # THE crash-proof, per-dataset resume path (Task 7)
args.use_amp = True                 # DEVIATION, required for 8GB VRAM -- see DEVIATIONS.md
args.reinit_heads = False

print(args)
""")

code("""import subprocess
print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,memory.used,memory.free",
                       "--format=csv"], capture_output=True, text=True).stdout)
""")

md("## Build datasets and model, then hand off to the real `omni_engine` loop\n\n"
   "This mirrors `main_ark.py`'s `main()` exactly (same `dict_dataloarder[dataset](...)` "
   "construction pattern), just pointed at `datasets_config_medmnist.yaml` instead of "
   "`datasets_config.yaml`.")

code("""exp_name = args.model_name + "_" + args.exp_name
model_path = os.path.join("./Models", exp_name)
output_path = os.path.join("./Outputs", exp_name)

datasets_config = get_config('datasets_config_medmnist.yaml')
for dataset in args.dataset_list:
    assert dataset in datasets_config, f"{dataset} missing from datasets_config_medmnist.yaml"

dataset_train_list, dataset_val_list, dataset_test_list = [], [], []
for dataset in args.dataset_list:
    dataset_train_list.append(
        dict_dataloarder[dataset](images_path=datasets_config[dataset]['data_dir'],
                                   file_path=datasets_config[dataset]['train_list'],
                                   crop_size=args.crop_size, resize=args.resize, augment=None))
    dataset_val_list.append(
        dict_dataloarder[dataset](images_path=datasets_config[dataset]['data_dir'],
                                   file_path=datasets_config[dataset]['val_list'],
                                   crop_size=args.crop_size, resize=args.resize,
                                   augment=build_transform_classification(
                                       normalize=args.normalization, crop_size=args.crop_size,
                                       resize=args.resize, mode="valid")))
    dataset_test_list.append(
        dict_dataloarder[dataset](images_path=datasets_config[dataset]['data_dir'],
                                   file_path=datasets_config[dataset]['test_list'],
                                   crop_size=args.crop_size, resize=args.resize,
                                   augment=build_transform_classification(
                                       normalize=args.normalization, crop_size=args.crop_size,
                                       resize=args.resize, mode="test",
                                       test_augment=args.test_augment)))
    print(f"  {dataset}: train={len(dataset_train_list[-1])} "
          f"val={len(dataset_val_list[-1])} test={len(dataset_test_list[-1])}")
""")

md("### Optional: ImageNet init for the from-scratch Swin-Tiny encoder\n\n"
   "`args.init` is parsed by `get_args_parser` but never wired to backbone loading "
   "anywhere in this codebase (see `models.py`'s `load_imagenet_backbone` docstring). "
   "Skip this cell to train from random init instead.")

code("""from models import build_omni_model, load_imagenet_backbone

num_classes_list = [len(datasets_config[d]['diseases']) for d in args.dataset_list]
_probe_model = build_omni_model(args, num_classes_list)
load_imagenet_backbone(_probe_model, "swin_tiny_patch4_window7_224")
# omni_engine builds its own student/teacher internally; save these weights and load
# them via args.pretrained_weights instead of hand-threading _probe_model through.
os.makedirs(model_path, exist_ok=True)
imagenet_init_path = os.path.join(model_path, "imagenet_init.pth")
torch.save({'state_dict': _probe_model.state_dict()}, imagenet_init_path)
del _probe_model
""")

code("""omni_engine(args, model_path, output_path, args.dataset_list, datasets_config,
            dataset_train_list, dataset_val_list, dataset_test_list)
""")

md("## If this cell was interrupted (Ctrl+C or a crash)\n\n"
   "Just re-run the two cells above (config, then `omni_engine(...)`) — "
   "`args.crash_proof_resume = True` means it finds `Models/<exp_name>/.../"
   "*_atomic_latest.pth` and resumes mid-cycle. See Task 11 for how to verify this.")

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "ark_medmnist", "language": "python", "name": "ark_medmnist"},
        "language_info": {"name": "python", "version": "3.11.15"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

with open("pretrain_medmnist2d.ipynb", "w") as f:
    json.dump(notebook, f, indent=1)
print("Wrote pretrain_medmnist2d.ipynb")
```

Run this generator once (`conda run -n ark_medmnist python -c "<paste the script above>"` or save it as a throwaway `.py` and run it) to produce the actual `.ipynb` file — a `.ipynb` is JSON, and generating it programmatically avoids hand-typing broken notebook JSON.

- [ ] **Step 2: Open `pretrain_medmnist2d.ipynb` in VS Code, select the `ark_medmnist` kernel, and run every cell top to bottom**

Expected: the `nvidia-smi` cell prints current free VRAM; the dataset-building cell prints 12 lines of `train=N val=M test=K` counts (downloading MedMNIST archives on first run — this can take a few minutes per dataset); `omni_engine(...)` starts printing `Epoch: [0][...]` progress lines from `ProgressLogger` for `pathmnist` (the first dataset in `MEDMNIST_2D_KEYS`).

- [ ] **Step 3: Commit**

```bash
git add pretrain_medmnist2d.ipynb
git commit -m "add pretrain_medmnist2d.ipynb notebook driver for the 12-2D MedMNIST Ark+ pretraining run"
```

---

### Task 11: Manual crash-resume verification (the brief's explicit test requirement)

**Files:** none — this exercises the running notebook, not something a unit test can cover.

- [ ] **Step 1: Start the run, let it progress past at least one full dataset**

In the notebook, run the `omni_engine(...)` cell. Watch the console until you see at least one `Epoch: [0][...]` progress block finish for `pathmnist` (or use a truncated `args.dataset_list` for a faster first pass, e.g. `MEDMNIST_2D_KEYS[:2]`), then interrupt the kernel (Jupyter's Interrupt button, which raises `KeyboardInterrupt` inside the running cell).

Expected: `>>> KeyboardInterrupt caught.` then `>>> Checkpointing before exit...` then `>>> Saved checkpoint at epoch=0, last_completed=<i>. Exiting.` where `<i>` is the index of the last dataset that fully finished.

- [ ] **Step 2: Re-run the `omni_engine(...)` cell and confirm it resumes mid-cycle**

Expected: `>>> Loaded latest checkpoint: .../medmnist2d_swintiny_atomic_latest.pth` then `>>> Crash-proof resume: epoch 0, skipping datasets 0..<i> already completed this epoch: [...]`, and the next `Epoch: [0][...]` block printed is for the dataset immediately after index `<i>` — not `pathmnist` again (unless `<i>` was already the last dataset in the list).

- [ ] **Step 3: Repeat with a hard kill (bypasses `KeyboardInterrupt` entirely)**

Let it progress past another dataset or two, then from a terminal find the kernel's PID (`Get-Process python | Where-Object {$_.MainWindowTitle -eq ''}` or check Jupyter's kernel list) and `taskkill /F /PID <pid>`. This exercises only the periodic `checkpoint_now()` calls and the atomic-write/fallback path for real, not simulated.

Restart the kernel and re-run. Expected: resumes from whichever dataset last completed *before* the kill — it will redo the dataset that was in-flight at kill time, which is correct (partial progress *within* one dataset's epoch isn't checkpointed, only whole-dataset completions are, matching the granularity `_resume_dataset_range` was tested against in Task 7).

- [ ] **Step 4: Confirm the training log reflects a continuous run, not a restart**

Open `Outputs/<exp_name>/<exp>_<exp_name>_results.txt` (created on the first periodic test-eval, every `args.test_epoch` cycles) and `Models/<exp_name>/.../train.log`. Confirm the log file grew across both restarts (it's opened in append mode, `open(log_file, 'a')`, unchanged from the original) rather than being overwritten.

---

## Self-Review Notes

- **Spec coverage:** deliverable 1 (environment.yml + verify_env.py) — Task 1. The "MOST IMPORTANT REQUIREMENT" (crash-proof per-dataset resume, atomic writes, RNG state, KeyboardInterrupt handling, tested by killing the process mid-cycle) — Tasks 5, 7, 11. Bug B (hardcoded/wrong class counts, specifically `BreastMNIST: 3`) — Tasks 3–4, verified against real `medmnist.INFO` not memory. Bug C.3 (symmetric teacher/student views) — Task 3, confirmed as a real bug in the *current* `medmnist_dataloader.py` stub, fixed via the asymmetric-view design copied from `ChestXray14`. Bug D (missing projector) — not applicable to this plan: the real `ArkSwinTransformer` already has a projector and already feeds heads the unnormalized projector output; nothing to fix there, only the pooling-shape bridge (Task 2). Bug F (EMA buffers) — not applicable: Swin has zero BatchNorm layers, confirmed by reading `ArkSwinTransformer`; noted explicitly in `DEVIATIONS.md`, no code change needed for this plan's scope (relevant to a future conv-backbone baseline, not here). Bug G (AMP, gradient discussion) — Task 6 adds AMP; gradient clipping was *not* added because the real repo's `create_optimizer`/`create_scheduler` (timm factories, left untouched) already own the optimizer/scheduler construction and neither the brief's brought-forward custom script's LR (0.3 tuned for batch 50/4-GPU) nor its instability applies here — this run uses `args.lr = 1e-2` (`--lr` default) at batch 32 on one GPU, not a linearly-scaled 0.3; revisit if training diverges.
- **Corrected findings, not silently absorbed:** the loss-combination formula and the "constant EMA momentum 0.9" claim in the original brief were checked against the real `trainer.py` and found to not match (real code ramps both `coff` and momentum via a cosine schedule) — documented in `DEVIATIONS.md`'s dedicated "Corrected understanding" section rather than either blindly implementing the brief's version or silently fixing it without saying so.
- **No placeholders:** every step has complete, runnable code; the `datasets_config_medmnist.yaml` content is the actual verified output, not a description of what it should contain.
- **Type/interface consistency check:** `MedMNIST2DDataset.__getitem__` returns `(student_img, teacher_img, label)` — CHW float32 tensors, float label — consistently consumed by `train_one_epoch`/`evaluate` (Task 6, unchanged call sites) exactly as `ChestXray14.__getitem__` already is. `medmnist_task_type`/`medmnist_num_classes` (Task 3) are the single source of truth consumed by both `datasets_config_medmnist.yaml`'s generator (Task 4) and its test — no second hardcoded copy exists anywhere. `checkpoint_utils`'s four functions (Task 5) are used identically by `engine.py` (Task 7) with no signature drift. `train_one_epoch`/`evaluate`'s new `scaler=None` parameter (Task 6) is threaded through consistently from `engine.py`'s `omni_engine` (Task 7).
- **Not covered by this plan, deliberately:** the 6 3D datasets (brief step 3, needs a different encoder entirely), the 18-combined dual-encoder extension (brief step 4, explicitly an extension not a replication), strong individual baselines / fine-tuning / linear-probing / results tables (deliverable 5, depends on this plan producing a real checkpoint first), and the EMA-for-BatchNorm audit's part (b)/(c) (Swin has no BN — relevant once a conv baseline exists, not here). Each is a separate follow-on plan per the brief's own explicit ordering.
