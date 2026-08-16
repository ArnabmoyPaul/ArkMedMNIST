# 6-Dataset 3D Ark+ Pretraining + Downstream Fine-Tuning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Get a dedicated 3D-only Ark+ omni-pretraining run across all 6 MedMNIST-3D
datasets, then fine-tune and evaluate the resulting encoder on each dataset
separately, comparing test AUC against professor-supplied benchmarks.

**Architecture:** Reuse the already-built (but uncommitted) 3D infrastructure
(`ArkR3D`, `build_omni_model_3d`, `load_kinetics_backbone`, the BatchNorm-EMA fix,
the dual-encoder-capable `omni_engine`) as-is. Add one new pretraining driver
(`pretrain_medmnist3d_run.py`, calling `omni_engine` with an all-3D `dataset_list`)
and implement the never-built downstream fine-tuning pieces
(`trainer.train_downstream_epoch`, `models.load_finetune_backbone`,
`finetune_downstream_3d.py`) per the already-reviewed 2026-08-12 design (generic,
not 2D-specific, so it drives `ArkR3D` unchanged).

**Tech Stack:** PyTorch 2.13.0+cu130, torchvision 0.28.0+cu130 (`r3d_18`), timm
1.0.28, `medmnist` 3.0.2, conda env `ark_medmnist`.

## Global Constraints

- Env spec: `environment.yml` (python=3.11.15, torch==2.13.0+cu130,
  torchvision==0.28.0+cu130, timm==1.0.28, medmnist==3.0.2) — source of truth,
  not to be changed.
- The 6 dataset keys, exact casing (from `medmnist3d_dataloader.DATASET_MAP_3D`):
  `OrganMNIST3D`, `NoduleMNIST3D`, `AdrenalMNIST3D`, `FractureMNIST3D`,
  `VesselMNIST3D`, `SynapseMNIST3D`.
- Benchmark targets (`BENCHMARKS_3D` in `finetune_downstream_3d.py`):
  OrganMNIST3D=0.994, VesselMNIST3D=0.905, AdrenalMNIST3D=0.828,
  FractureMNIST3D=0.725, NoduleMNIST3D=0.875, SynapseMNIST3D=None (no target).
  Verdict: PASS (auc >= target), WITHIN_1PCT (target - auc <= 0.01),
  FAIL (otherwise), NO_TARGET (target is None).
- Pretraining defaults: `batch_size_3d=16`, `lr_3d=1e-3`, `momentum_3d=0.9`
  (already validated on this GPU by the abandoned 18-combined run),
  `pretrain_epochs=20`, `test_epoch=5`, `use_amp=True`, `crash_proof_resume=True`,
  `exp_name="medmnist3d_r3d18"`. `crop_size=112`/`resize=128`/`batch_size=32`
  still set even though unused (the idle 2D branch's `build_omni_model` reads
  `args.crop_size` unconditionally and would crash without it).
- Fine-tuning checkpoint key: the 3D-only run's teacher weights live under
  `'teacher_3d'` in the checkpoint dict, NOT `'teacher'` (that key holds the
  idle/unused 2D model). `load_finetune_backbone(model, checkpoint, key="teacher_3d")`.
- Fine-tuning defaults: `--lr 1e-3`, `--epochs 20`, `--patience 5`,
  `--batch_size 16` (matches `batch_size_3d`), full fine-tune (no linear probe),
  `--checkpoint` is a **required** CLI arg (no hardcoded default — unlike the 2D
  plan's `DEFAULT_CHECKPOINT`, the ideal epoch isn't known until after Task 7's
  real pretraining run completes and its `epoch_metrics.jsonl` is inspected).
- Test file style: plain `test_*` functions, no pytest fixtures/classes,
  `if __name__ == "__main__":` block, prints `"<file>: all checks passed"` —
  matches every existing test file (`test_ema_buffer.py`, `test_trainer_amp.py`, etc).
- Out of scope: the abandoned 18-combined joint run and its logs; any change to
  2D-only behavior; guarding `omni_engine` against building the idle 2D model
  (flagged, not built — `# ponytail: idle unused 2D model wastes ~300MB VRAM when
  dataset_list is 3D-only, guard it only if this actually OOMs`); an open-ended
  hyperparameter search tool (Task 8's retry procedure is fixed and bounded).

---

## File Structure

- Recreate: conda env `ark_medmnist` (no repo files)
- Commit as-is (already written, uncommitted): `dataloader.py`, `medmnist_dataloader.py`,
  `trainer.py`, `models.py`, `medmnist3d_dataloader.py`, `engine.py`, `DEVIATIONS.md`,
  `pretrain_medmnist2d.ipynb`, `test_models_medmnist.py`, `test_trainer_amp.py`,
  `test_ema_buffer.py`, `test_medmnist3d_dataloader.py`, `test_models_medmnist3d.py`
- Create: `pretrain_medmnist3d_run.py`
- Modify: `trainer.py` — add `train_downstream_epoch()`
- Create: `test_trainer_downstream.py`
- Modify: `models.py` — add `load_finetune_backbone()`
- Modify: `test_models_medmnist.py` — add tests for `load_finetune_backbone()`
- Create: `finetune_downstream_3d.py`
- Create: `test_finetune_downstream_3d.py`

---

### Task 1: Recreate the `ark_medmnist` conda environment

**Files:** none (environment operation only)

**Interfaces:** none — this unblocks every later task, which all assume a working
`ark_medmnist` env with torch/CUDA available.

- [ ] **Step 1: Confirm the env is actually empty (not a false alarm)**

Run: `powershell -Command "(Get-ChildItem 'D:\anaconda\envs\ark_medmnist\conda-meta' -Filter '*.json').Count"`
Expected: `0` (confirmed during planning — `conda-meta/history` shows a full package
removal despite training logs proving the env worked as of Aug 15 23:23).

- [ ] **Step 2: Remove the empty env shell and recreate from the pinned spec**

Run:
```
conda env remove -n ark_medmnist -y
conda env create -f "D:\ark +\environment.yml"
```

- [ ] **Step 3: Verify the stack imports and CUDA is visible**

Run: `conda run -n ark_medmnist python -c "import torch, torchvision, timm, medmnist; print(torch.__version__, torch.cuda.is_available())"`
Expected: `2.13.0+cu130 True`

If CUDA is `False`, stop and report — do not proceed to GPU-bound tasks on CPU.

---

### Task 2: Verify and land the pre-existing 3D/AMP WIP

**Files:** `dataloader.py`, `medmnist_dataloader.py`, `trainer.py`, `models.py`,
`medmnist3d_dataloader.py`, `engine.py`, `DEVIATIONS.md`, `pretrain_medmnist2d.ipynb`,
`test_models_medmnist.py`, `test_trainer_amp.py`, `test_ema_buffer.py`,
`test_medmnist3d_dataloader.py`, `test_models_medmnist3d.py`

**Interfaces:**
- Consumes: nothing new — all code in these files already exists in the working
  tree (uncommitted modifications + untracked new files), written in an earlier
  session. This task verifies it actually works now that Task 1 restored the env,
  and commits it.
- Produces: `ArkR3D`, `build_omni_model_3d`, `load_kinetics_backbone` (`models.py`);
  `DATASET_MAP_3D`, `DATASETS_CONFIG_3D`, `MedMNIST3DWrapper` (`medmnist3d_dataloader.py`);
  BatchNorm-aware `ema_update_teacher`, `test_classification(..., is_3d=)` (`trainer.py`);
  dual-encoder `omni_engine` (`engine.py`) — all consumed directly by Tasks 3-6.

- [ ] **Step 1: Run the full existing test suite**

Run (from `D:\ark +`, in the `ark_medmnist` env):
```
conda run -n ark_medmnist python test_checkpoint_utils.py
conda run -n ark_medmnist python test_compat_albumentations.py
conda run -n ark_medmnist python test_datasets_config_medmnist.py
conda run -n ark_medmnist python test_engine_resume_indexing.py
conda run -n ark_medmnist python test_get_args_parser_argv.py
conda run -n ark_medmnist python test_medmnist_dataloader.py
conda run -n ark_medmnist python test_models_medmnist.py
conda run -n ark_medmnist python test_trainer_amp.py
```
Expected: every file prints its own `"<file>: all checks passed"` line, no tracebacks.

- [ ] **Step 2: Run the three new (untracked) 3D test files**

Run:
```
conda run -n ark_medmnist python test_ema_buffer.py
conda run -n ark_medmnist python test_medmnist3d_dataloader.py
conda run -n ark_medmnist python test_models_medmnist3d.py
```
Expected: same — all three print their pass lines.

- [ ] **Step 3: Fix anything that fails**

If any test fails, read the failure, fix the bug in the relevant source file (not
the test, unless the test itself is wrong — check both), and re-run until Steps 1-2
are all green. This is real verification, not an assumption — the env has been
broken since before this plan started, so none of this WIP has actually been
exercised in its current form yet.

- [ ] **Step 4: Commit the WIP in 6 logical groups**

```bash
git add DEVIATIONS.md dataloader.py medmnist_dataloader.py
git commit -m "$(cat <<'EOF'
fix Windows multiprocessing spawn picklability: functools.partial over local classes/lambdas

DataLoader(num_workers>0) on Windows re-imports worker arguments via spawn,
which requires them picklable. Hit in medmnist_dataloader.py's per-dataset
classes and dataloader.py's TenCrop test-augment lambdas. No change to what
the transforms compute.
EOF
)"

git add trainer.py test_ema_buffer.py test_trainer_amp.py
git commit -m "$(cat <<'EOF'
fix ema_update_teacher to EMA BatchNorm buffers, not just parameters; add is_3d flag to test_classification

BatchNorm running_mean/running_var live in .buffers(), not .parameters() --
without this the teacher's BN stats never move, invisible on the 2D-only run
(no BatchNorm there) but a real correctness bug for the upcoming 3D ArkR3D
encoder. is_3d makes test_classification's TTA-vs-volume shape branch
explicit instead of shape-sniffing (both are rank-5 tensors).
EOF
)"

git add models.py test_models_medmnist.py test_models_medmnist3d.py
git commit -m "$(cat <<'EOF'
add ArkR3D/build_omni_model_3d/load_kinetics_backbone: 3D encoder counterpart to ArkSwinTransformer

Wraps torchvision r3d_18 (1-channel stem, Kinetics-init via channel-averaged
stem weights) behind the same forward(x, head_n) contract trainer.py already
drives. Also fixes load_imagenet_backbone to filter shape-mismatched keys
(crop_size<224 shrinks the last Swin stage's window, changing its
relative_position_bias_table shape) instead of relying on strict=False alone.
EOF
)"

git add medmnist3d_dataloader.py test_medmnist3d_dataloader.py
git commit -m "$(cat <<'EOF'
fix MedMNIST3DWrapper label encoding to one-hot; add deterministic teacher view

trainer.py does targets.float() unconditionally -- a bare class index
silently mismatched CrossEntropyLoss's (B, num_classes) expectation. Reuses
medmnist_dataloader's _to_onehot_or_multihot. Teacher view is now a separate
non-augmenting Augment3D instance, matching Ark+'s stable-teacher-signal design.
EOF
)"

git add engine.py
git commit -m "$(cat <<'EOF'
add dual-encoder support to omni_engine: route 2D/3D datasets to separate student/teacher pairs

is_3d_list/_route dispatch each dataset to build_omni_model (Swin) or
build_omni_model_3d (ArkR3D) with its own optimizer/scheduler, since the two
encoders don't share weights or a training signal. Also: hash-based exp
naming (dataset-name concatenation exceeded Windows MAX_PATH at 18 datasets),
per-epoch JSONL metrics log, per-modality val-loss/AUC reporting.
EOF
)"

git add pretrain_medmnist2d.ipynb
git commit -m "add test_epoch=5 config and fix missing pretrained_weights wiring in the 2D notebook driver"
```

- [ ] **Step 5: Confirm clean tree**

Run: `git status`
Expected: only the still-untracked, out-of-scope files remain (`pretrain_medmnist18_run.py`,
`Models/`, `*.log`, `epoch_report.html`, `MedArk3D_v5_colab.ipynb`,
`_run_12_2d_epoch15.ipynb`) — none of them touched by this plan.

---

### Task 3: `pretrain_medmnist3d_run.py`

**Files:**
- Create: `pretrain_medmnist3d_run.py`

**Interfaces:**
- Consumes: `omni_engine` (`engine.py`, Task 2), `build_omni_model_3d`,
  `load_kinetics_backbone` (`models.py`, Task 2), `MedMNIST3DWrapper`,
  `DATASET_MAP_3D`, `DATASETS_CONFIG_3D` (`medmnist3d_dataloader.py`, Task 2),
  `get_args_parser` (`main_ark.py`, unchanged).
- Produces: `main(argv=None)` — no return value, runs `omni_engine` to completion
  or until interrupted. Consumed by Task 7 (the real run) as
  `python pretrain_medmnist3d_run.py`.

- [ ] **Step 1: Write the driver**

Create `pretrain_medmnist3d_run.py`:

```python
def main():
    import sys, os
    sys.argv = ['pretrain_medmnist3d']  # keep optparse away from Jupyter's own -f <kernel.json>

    from main_ark import get_args_parser
    from medmnist3d_dataloader import MedMNIST3DWrapper, DATASET_MAP_3D, DATASETS_CONFIG_3D
    from engine import omni_engine

    args = get_args_parser(argv=[])

    args.model_name = "swin_tiny"             # only matters for the idle unused 2D branch
    args.dataset_list = list(DATASET_MAP_3D)   # all 6: Organ/Nodule/Adrenal/Fracture/Vessel/Synapse
    args.crop_size = 112                       # unused (no 2D datasets) but build_omni_model
    args.resize = 128                          # reads args.crop_size unconditionally
    args.batch_size = 32                       # 2D batch size, unused
    args.batch_size_3d = 16                    # validated on this GPU by the 18-combined run
    args.workers = 4
    args.pretrain_epochs = 20                  # matches the 2D-12 run's precedent
    args.test_epoch = 5                        # full test-set AUC pass every 5 epochs -- this
                                                # pass is 2-3x slower than training (TTA on
                                                # student+teacher over all 6 test sets)
    args.momentum_teacher = 0.9
    args.ema_mode = "epoch"
    args.exp_name = "medmnist3d_r3d18"
    args.projector_features = 512
    args.use_mlp = False
    args.pretrained_weights = None             # 2D branch stays random-init, never trained here
    args.pretrained_weights_3d = None          # Kinetics init applied separately below
    args.opt = "momentum"
    args.lr = 1e-2                             # 2D encoder lr, unused
    args.momentum = 0.9
    args.lr_3d = 1e-3                          # validated on this GPU by the 18-combined run
    args.momentum_3d = 0.9
    args.weight_decay = 1e-4
    args.resume = False                        # original epoch-only resume path -- off, using crash_proof_resume
    args.crash_proof_resume = True             # this GPU has a documented crash history
    args.use_amp = True                        # required for 8GB VRAM -- see DEVIATIONS.md
    args.reinit_heads = False

    print(args)

    import subprocess
    print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,memory.used,memory.free",
                           "--format=csv"], capture_output=True, text=True).stdout)

    exp_name = args.model_name + "_" + args.exp_name
    model_path = os.path.join("./Models", exp_name)
    output_path = os.path.join("./Outputs", exp_name)

    datasets_config = DATASETS_CONFIG_3D

    dataset_train_list, dataset_val_list, dataset_test_list = [], [], []
    for dataset in args.dataset_list:
        dataset_train_list.append(MedMNIST3DWrapper(dataset, split='train'))
        dataset_val_list.append(MedMNIST3DWrapper(dataset, split='val'))
        dataset_test_list.append(MedMNIST3DWrapper(dataset, split='test'))
        print(f"  {dataset}: train={len(dataset_train_list[-1])} "
              f"val={len(dataset_val_list[-1])} test={len(dataset_test_list[-1])}")

    import torch
    from models import build_omni_model_3d, load_kinetics_backbone

    num_classes_list_3d = [len(datasets_config[d]['diseases']) for d in args.dataset_list]

    os.makedirs(model_path, exist_ok=True)

    # omni_engine builds its own student/teacher internally; save the probe's
    # Kinetics-initialized weights and load them via args.pretrained_weights_3d
    # instead of hand-threading the probe model through.
    _probe_3d = build_omni_model_3d(args, num_classes_list_3d)
    load_kinetics_backbone(_probe_3d)
    kinetics_init_path = os.path.join(model_path, "kinetics_init.pth")
    torch.save({'state_dict': _probe_3d.state_dict()}, kinetics_init_path)
    del _probe_3d
    args.pretrained_weights_3d = kinetics_init_path

    omni_engine(args, model_path, output_path, args.dataset_list, datasets_config,
                dataset_train_list, dataset_val_list, dataset_test_list)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-test with 1 epoch**

Temporarily edit the line `args.pretrain_epochs = 20` to `args.pretrain_epochs = 1`,
then run:

Run: `conda run -n ark_medmnist python pretrain_medmnist3d_run.py`

Expected:
- No traceback.
- Console shows all 6 dataset sizes printed, then one full epoch's worth of
  `Epoch: [0][...]` lines for each of the 6 datasets, then
  `Epoch 0000: avg_val_loss ... (2D ..., 3D ...)`.
- `Models/swin_tiny_medmnist3d_r3d18/Ark_Plus_6ds_<hash>/medmnist3d_r3d18/` contains
  a checkpoint file.

If this fails, fix the driver script directly and rerun until it passes.

- [ ] **Step 3: Revert to the real epoch count**

Edit `args.pretrain_epochs = 1` back to `args.pretrain_epochs = 20`.

- [ ] **Step 4: Commit**

```bash
git add pretrain_medmnist3d_run.py
git commit -m "$(cat <<'EOF'
add pretrain_medmnist3d_run.py: dedicated 6-dataset 3D-only Ark+ omni-pretraining driver

Parallels pretrain_medmnist2d_run.py's structure (own args namespace, direct
omni_engine call, no main_ark.py CLI parser). dataset_list is all 6 MedMNIST-3D
keys -- the 2D student/teacher omni_engine still builds internally stays idle
and unused (ponytail: not worth guarding out, wastes ~300MB VRAM, see
DEVIATIONS.md/design spec).
EOF
)"
```

---

### Task 4: `train_downstream_epoch` in trainer.py

**Files:**
- Modify: `trainer.py`
- Test: `test_trainer_downstream.py`

**Interfaces:**
- Produces: `train_downstream_epoch(model, use_head_n, dataset, data_loader_train, device, criterion, optimizer, epoch, scaler=None) -> float` (average training loss for the epoch). Dataloader batches are `(samples, _, targets)` — the second (teacher-view) element is ignored, same as `evaluate()` already does.

- [ ] **Step 1: Write the failing test**

Create `test_trainer_downstream.py`:

```python
"""test_trainer_downstream.py — Run: python test_trainer_downstream.py"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from trainer import train_downstream_epoch


class _TinyArkLike(nn.Module):
    """Mimics ArkR3D/ArkSwinTransformer's forward(x, head_n) -> (feat, logits)
    contract with a trivial linear encoder, so this test has no GPU/timm
    dependency and works for either modality's model shape."""
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

Run: `conda run -n ark_medmnist python test_trainer_downstream.py`
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

Run: `conda run -n ark_medmnist python test_trainer_downstream.py`
Expected: `test_trainer_downstream.py: all checks passed`

- [ ] **Step 5: Commit**

```bash
git add trainer.py test_trainer_downstream.py
git commit -m "add train_downstream_epoch: plain supervised fine-tuning step, no distillation"
```

---

### Task 5: `load_finetune_backbone` in models.py

**Files:**
- Modify: `models.py`
- Modify: `test_models_medmnist.py`

**Interfaces:**
- Consumes: any model with `.load_state_dict` and `omni_heads.*`-prefixed head
  keys (`ArkSwinTransformer` or `ArkR3D`, both already satisfy this).
- Produces: `load_finetune_backbone(model, checkpoint_path, key='teacher') -> torch.nn.modules.module._IncompatibleKeys`.

- [ ] **Step 1: Write the failing test**

Add to `test_models_medmnist.py` (add `import os` to the existing import block at
the top if not already present, then append these two tests before the
`if __name__ == "__main__":` block):

```python
def test_load_finetune_backbone_transfers_encoder_not_heads():
    pretrained = ArkSwinTransformer([9, 2, 5], projector_features=None, use_mlp=False,
                                     patch_size=4, window_size=7, embed_dim=96,
                                     depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24))
    ckpt_path = "test_fake_checkpoint_heads.pth.tar"
    torch.save({"teacher": pretrained.state_dict()}, ckpt_path)

    try:
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

Update the import line to include `load_finetune_backbone`:
```python
from models import ArkSwinTransformer, build_omni_model, load_imagenet_backbone, load_finetune_backbone
```

Update the `if __name__ == "__main__":` block at the bottom to also call both new
tests (add alongside whatever Task 2 already landed there).

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n ark_medmnist python test_models_medmnist.py`
Expected: `ImportError: cannot import name 'load_finetune_backbone' from 'models'`

- [ ] **Step 3: Implement `load_finetune_backbone` in models.py**

Add to `models.py`, after `load_imagenet_backbone`:

```python
def load_finetune_backbone(model, checkpoint_path, key='teacher'):
    """Loads a single-task-head model's encoder from a multi-head omni-pretraining
    checkpoint. omni_heads keys are always dropped before loading: the
    checkpoint's omni_heads.0 is whichever dataset was first in the ORIGINAL
    pretraining dataset_list, not this model's target dataset -- loading it
    unfiltered would crash on a shape mismatch, or worse, silently splice in
    the wrong dataset's head whenever class counts happen to match
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

Run: `conda run -n ark_medmnist python test_models_medmnist.py`
Expected: `test_models_medmnist.py: all checks passed`

- [ ] **Step 5: Commit**

```bash
git add models.py test_models_medmnist.py
git commit -m "add load_finetune_backbone: transfer encoder from an omni checkpoint, strip heads"
```

---

### Task 6: `finetune_downstream_3d.py` driver script

**Files:**
- Create: `finetune_downstream_3d.py`
- Test: `test_finetune_downstream_3d.py`

**Interfaces:**
- Consumes: `train_downstream_epoch` (Task 4), `load_finetune_backbone` (Task 5),
  `build_omni_model_3d`, `save_checkpoint` (`models.py`), `evaluate`,
  `test_classification` (`trainer.py`), `MedMNIST3DWrapper`, `DATASET_MAP_3D`,
  `DATASETS_CONFIG_3D` (`medmnist3d_dataloader.py`), `metric_AUROC` (`utils.py`).
- Produces: `_early_stop(val_losses, patience) -> bool`,
  `_compare_to_benchmark(auc, target) -> str`, `parse_args(argv=None) -> Namespace`,
  `BENCHMARKS_3D: dict`, `main(argv=None) -> dict`.

- [ ] **Step 1: Write the failing test**

Create `test_finetune_downstream_3d.py`:

```python
"""test_finetune_downstream_3d.py — Run: python test_finetune_downstream_3d.py"""
from finetune_downstream_3d import _early_stop, _compare_to_benchmark, parse_args, BENCHMARKS_3D


def test_no_stop_while_still_improving():
    assert not _early_stop([0.9, 0.8, 0.7, 0.6, 0.5], patience=2)


def test_no_stop_before_patience_window_elapses():
    assert not _early_stop([0.5, 0.6, 0.7], patience=3)


def test_stops_after_patience_epochs_without_improvement():
    assert _early_stop([0.9, 0.4, 0.5, 0.6, 0.7], patience=3)


def test_stops_exactly_at_patience_boundary():
    assert _early_stop([0.5, 0.6, 0.7], patience=2)


def test_compare_pass_when_auc_meets_or_beats_target():
    assert _compare_to_benchmark(0.994, 0.994) == "PASS"
    assert _compare_to_benchmark(0.999, 0.994) == "PASS"


def test_compare_within_1pct_when_short_by_at_most_one_point():
    assert _compare_to_benchmark(0.984, 0.994) == "WITHIN_1PCT"


def test_compare_fail_when_short_by_more_than_one_point():
    assert _compare_to_benchmark(0.90, 0.994) == "FAIL"


def test_compare_no_target_for_synapse():
    assert _compare_to_benchmark(0.80, None) == "NO_TARGET"
    assert BENCHMARKS_3D["SynapseMNIST3D"] is None


def test_parse_args_requires_checkpoint():
    try:
        parse_args(["--dataset", "OrganMNIST3D"])
        assert False, "expected SystemExit for missing required --checkpoint"
    except SystemExit:
        pass


def test_parse_args_defaults():
    args = parse_args(["--dataset", "OrganMNIST3D", "--checkpoint", "fake.pth.tar"])
    assert args.dataset == "OrganMNIST3D"
    assert args.checkpoint == "fake.pth.tar"
    assert args.epochs == 20
    assert args.lr == 1e-3
    assert args.patience == 5


if __name__ == "__main__":
    test_no_stop_while_still_improving()
    test_no_stop_before_patience_window_elapses()
    test_stops_after_patience_epochs_without_improvement()
    test_stops_exactly_at_patience_boundary()
    test_compare_pass_when_auc_meets_or_beats_target()
    test_compare_within_1pct_when_short_by_at_most_one_point()
    test_compare_fail_when_short_by_more_than_one_point()
    test_compare_no_target_for_synapse()
    test_parse_args_requires_checkpoint()
    test_parse_args_defaults()
    print("test_finetune_downstream_3d.py: all checks passed")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n ark_medmnist python test_finetune_downstream_3d.py`
Expected: `ModuleNotFoundError: No module named 'finetune_downstream_3d'`

- [ ] **Step 3: Implement `finetune_downstream_3d.py`**

Create `finetune_downstream_3d.py`:

```python
import argparse
import json
import os
import time
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader

from medmnist3d_dataloader import DATASET_MAP_3D, DATASETS_CONFIG_3D, MedMNIST3DWrapper
from models import build_omni_model_3d, load_finetune_backbone, save_checkpoint
from trainer import train_downstream_epoch, evaluate, test_classification
from utils import metric_AUROC

BENCHMARKS_3D = {
    "OrganMNIST3D": 0.994,
    "VesselMNIST3D": 0.905,
    "AdrenalMNIST3D": 0.828,
    "FractureMNIST3D": 0.725,
    "NoduleMNIST3D": 0.875,
    "SynapseMNIST3D": None,  # no target given -- report only
}


def _compare_to_benchmark(auc, target):
    """PASS if auc meets/beats target, WITHIN_1PCT if short by <=0.01,
    FAIL otherwise. NO_TARGET when the benchmark table has no entry."""
    if target is None:
        return "NO_TARGET"
    if auc >= target:
        return "PASS"
    if target - auc <= 0.01:
        return "WITHIN_1PCT"
    return "FAIL"


def _early_stop(val_losses, patience):
    """True once val_losses hasn't set a new best in the last `patience` epochs."""
    if len(val_losses) <= patience:
        return False
    best_idx = min(range(len(val_losses)), key=lambda i: val_losses[i])
    return best_idx <= len(val_losses) - 1 - patience


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Downstream fine-tune a 3D MedMNIST omni-pretrained checkpoint")
    p.add_argument("--dataset", required=True, choices=list(DATASET_MAP_3D))
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--workers", type=int, default=4)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds_cfg = DATASETS_CONFIG_3D[args.dataset]
    diseases = ds_cfg["diseases"]
    multiclass = ds_cfg["task_type"] == "multi-class classification"
    criterion = torch.nn.CrossEntropyLoss() if multiclass else torch.nn.BCEWithLogitsLoss()

    train_set = MedMNIST3DWrapper(args.dataset, split="train")
    val_set = MedMNIST3DWrapper(args.dataset, split="val")
    test_set = MedMNIST3DWrapper(args.dataset, split="test")

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=max(1, args.batch_size // 2), shuffle=False,
                              num_workers=args.workers, pin_memory=True)

    model_args = SimpleNamespace(projector_features=512, use_mlp=False, pretrained_weights_3d=None)
    model = build_omni_model_3d(model_args, num_classes_list=[len(diseases)])
    load_finetune_backbone(model, args.checkpoint, key="teacher_3d")
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

    y_test, p_test = test_classification(model, 0, test_loader, device, multiclass, is_3d=True)
    individual_auc = metric_AUROC(y_test, p_test, len(diseases))
    overall_auc = float(np.mean(individual_auc))
    target = BENCHMARKS_3D[args.dataset]
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

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n ark_medmnist python test_finetune_downstream_3d.py`
Expected: `test_finetune_downstream_3d.py: all checks passed`

- [ ] **Step 5: Commit**

```bash
git add finetune_downstream_3d.py test_finetune_downstream_3d.py
git commit -m "add finetune_downstream_3d.py: per-dataset 3D downstream fine-tuning driver with benchmark comparison"
```

- [ ] **Step 6: Real smoke test using Task 3's smoke checkpoint**

Use the 1-epoch checkpoint produced during Task 3 Step 2 (path under
`Models/swin_tiny_medmnist3d_r3d18/Ark_Plus_6ds_<hash>/medmnist3d_r3d18/`). OrganMNIST3D
has the fewest training steps of the 6 (61 batches at batch_size_3d=16, confirmed
from prior training logs), so it's the cheapest smoke test.

Run: `conda run -n ark_medmnist python finetune_downstream_3d.py --dataset OrganMNIST3D --checkpoint "<path to the Task 3 smoke checkpoint>" --epochs 1`

Expected:
- No traceback.
- Console shows `Loaded teacher_3d backbone from ... with msg: ...`, one
  `Epoch 0: train_loss=... val_loss=...` line, `Loaded best checkpoint from epoch 0 ...`,
  then `--- Overall test mAUC: <some 0-1 float> (target=0.994) -> ... ---`.
- `Models/finetune_OrganMNIST3D/best.pth.tar` and `Outputs/finetune_OrganMNIST3D/results.json` exist.
- This is a wiring smoke test only (checkpoint is from 1 epoch of pretraining) — the
  AUC does not need to be good, just present and in [0,1].

If this fails, fix the driver directly (no test to update — verifying glue code) and
rerun until it passes. Delete the smoke-test `Models/finetune_OrganMNIST3D/` and
`Outputs/finetune_OrganMNIST3D/` artifacts afterward (or leave them — Task 8 overwrites
them with the real run's results either way).

---

### Task 7: Run the real 20-epoch 3D pretraining

**Files:** none — operational task, running Task 3's driver for real.

- [ ] **Step 1: Launch the real pretraining run in the background**

Run (background, this takes hours):
`conda run -n ark_medmnist python pretrain_medmnist3d_run.py > train_3d.log 2>&1`

- [ ] **Step 2: Monitor to completion or crash**

Tail `train_3d.log` and watch for `Epoch 0019:` (the last of 20 epochs, 0-indexed)
or a traceback. If the process dies (this GPU has a crash history), `omni_engine`'s
`crash_proof_resume` machinery means simply re-running the same command resumes from
the last completed dataset within the last completed epoch — do this until epoch 19
finishes cleanly.

- [ ] **Step 3: Identify the best checkpoint**

Run: `python -c "import json; [print(l) for l in open(r'Models/swin_tiny_medmnist3d_r3d18/Ark_Plus_6ds_<hash>/medmnist3d_r3d18/epoch_metrics.jsonl')]"`
(fill in `<hash>` from the actual directory created in Step 1)

Find the epoch with the highest overall student mAUC across the 6 datasets (printed
to `train_3d.log` as `--- Overall student mAUC: ... ---` every `test_epoch` interval,
and recorded in `epoch_metrics.jsonl`'s `mAUC` lines). Use that epoch's numbered
checkpoint (`.../Ark_Plus_6ds_<hash><epoch>.pth.tar`) as the checkpoint for Task 8 —
matches the 2D run's precedent of picking the mAUC-peak epoch rather than always the
literal last one.

---

### Task 8: Downstream fine-tune all 6 datasets, tune to the benchmarks, report

**Files:** none — operational task, running Task 6's driver repeatedly.

- [ ] **Step 1: Fine-tune all 6 datasets against the Task 7 checkpoint**

For each of `OrganMNIST3D NoduleMNIST3D AdrenalMNIST3D FractureMNIST3D VesselMNIST3D SynapseMNIST3D`:

Run: `conda run -n ark_medmnist python finetune_downstream_3d.py --dataset <key> --checkpoint "<Task 7 best checkpoint path>"`

Collect each run's final `--- Overall test mAUC: ... -> <verdict> ---` line and its
`Outputs/finetune_<key>/results.json`.

- [ ] **Step 2: Retry any dataset with verdict FAIL or WITHIN_1PCT**

Per dataset needing a retry, up to 3 attempts total, one change per attempt:
1. Lower `--lr` to `3e-4` (same `--checkpoint`, same `--epochs 20`).
2. If still short: lower `--lr` to `1e-4`.
3. If still short: keep `--lr 1e-4`, raise `--epochs` to `40`.

Re-run: `conda run -n ark_medmnist python finetune_downstream_3d.py --dataset <key> --checkpoint "<Task 7 best checkpoint path>" --lr <value> --epochs <value>`

Stop retrying a dataset as soon as it reaches PASS.

- [ ] **Step 3: If datasets are still short after Step 2, extend pretraining once**

For any dataset still FAIL/WITHIN_1PCT after 3 fine-tune attempts: resume Task 7's
pretraining run for additional epochs (`crash_proof_resume` supports resuming past
a completed run by raising `args.pretrain_epochs` in `pretrain_medmnist3d_run.py`
and rerunning — `omni_engine`'s `args.resume` epoch-checkpoint path picks up from
the last saved epoch). Re-run Step 1-2 for the still-short datasets only against
the new final checkpoint.

- [ ] **Step 4: Report the final results table**

For each of the 6 datasets: dataset name, benchmark target, achieved test mAUC,
verdict, number of fine-tune attempts taken. Pull the numbers from each dataset's
final `Outputs/finetune_<key>/results.json` (`test_overall_mAUC`, `verdict`).

---

## Self-Review Notes

- **Spec coverage:** all four spec phases covered — Task 1 (env repair), Task 2
  (land WIP), Task 3 (pretraining driver), Tasks 4-6 (downstream fine-tuning driver,
  correcting the spec's earlier wrong assumption that `finetune_downstream.py`
  already existed), Tasks 7-8 (the bounded run + tune-to-benchmark procedure from
  the spec's "Autonomous tuning procedure" section, including the 3-attempt retry
  cap and single pretraining-extension round).
- **Type/signature consistency:** `train_downstream_epoch`'s signature (Task 4)
  matches its Task 6 call site exactly. `load_finetune_backbone`'s signature
  (Task 5) matches its Task 6 call site (`model, args.checkpoint, key="teacher_3d"`).
  `test_classification(model, use_head_n, data_loader_test, device, multiclass, is_3d)`
  call in Task 6 matches the signature landed by Task 2 (`trainer.py`'s uncommitted
  diff, verified in Task 2 Step 1-2). `build_omni_model_3d(args, num_classes_list)`
  calls in Tasks 3 and 6 both supply `args.projector_features`/`args.use_mlp` and
  (via getattr default or explicit `None`) `pretrained_weights_3d`, matching Task 2's
  landed signature.
- **No placeholders:** every step has complete, runnable code. Task 8's checkpoint
  paths are intentionally left as `<...>` placeholders to fill in from real Task 7
  output (not knowable until that run completes) — this is a plan-execution-time
  value, not a missing design decision, consistent with `--checkpoint` being a
  required (not defaulted) CLI arg per the Global Constraints.
