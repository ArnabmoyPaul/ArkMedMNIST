# 6-Dataset 3D Ark+ Replication: Pretraining + Downstream Fine-Tuning

## Goal

Build a dedicated omni-pretraining run over all 6 MedMNIST-3D datasets (OrganMNIST3D,
NoduleMNIST3D, AdrenalMNIST3D, FractureMNIST3D, VesselMNIST3D, SynapseMNIST3D),
mirroring the existing 12-dataset 2D pipeline's structure and already-validated Ark+
mechanics (cyclic multi-dataset teacher-student, self-distillation loss, EMA), then
fine-tune and evaluate the resulting encoder separately on each 3D dataset. Compare
final test AUC against the professor-supplied targets:

| Dataset | Target AUC |
|---|---|
| OrganMNIST3D | 0.994 |
| VesselMNIST3D | 0.905 |
| AdrenalMNIST3D | 0.828 |
| FractureMNIST3D | 0.725 |
| NoduleMNIST3D | 0.875 |
| SynapseMNIST3D | none given — report only, not graded |

Success per dataset: meet or beat the target, or land within 1% of it.

## Context

The original Ark+ repo (jlianglab/Ark) trains 2D and 3D as two **separate**
pipelines (`train_ark_12datasets.py` vs `train_ark_3d_3datasets.py`), never jointly.
An earlier session in this repo built an 18-dataset **joint** 2D+3D run
(`pretrain_medmnist18_run.py`, dual Swin-tiny/R3D-18 encoders sharing one
`omni_engine` call) instead — a deviation from the paper's design, and it crashed
repeatedly (4+ crash/resume cycles, Aug 11-15) without ever completing. That path is
being abandoned in favor of a dedicated 3D-only run, matching the paper's actual
architecture.

Investigation before this design found the following already built, uncommitted,
and directly reusable for a 3D-only run (no joint-run baggage):

- `ArkR3D` / `build_omni_model_3d` / `load_kinetics_backbone` (`models.py`) — full
  3D encoder (torchvision `r3d_18`, 1-channel stem, Kinetics-init) behind the same
  `forward(x, head_n)` contract as `ArkSwinTransformer`. Unit-tested
  (`test_models_medmnist3d.py`).
- `medmnist3d_dataloader.py` — one-hot labels (fixes a real bug: a bare class index
  would silently mismatch `CrossEntropyLoss`'s `(B, num_classes)` expectation once
  `trainer.py` does `targets.float()`), deterministic teacher view. Unit-tested
  (`test_medmnist3d_dataloader.py`).
- `trainer.py`'s `ema_update_teacher` now EMAs BatchNorm running stats, not just
  parameters. This is a real correctness fix specifically for 3D: `ArkSwinTransformer`
  has no BatchNorm, so this bug was invisible on the 2D-only run; `ArkR3D` uses
  BatchNorm3d throughout, so without this fix the teacher's BN stats would freeze on
  whichever dataset trained first in epoch 0 and never move again. Covered by
  `test_ema_buffer.py`.
- `trainer.py`'s `test_classification(..., is_3d=)` — explicit flag instead of
  shape-sniffing (a 3D volume batch and a 2D 10-crop TTA batch are both rank-5
  tensors; inferring modality from `.dim()` would confuse them).
- `engine.py`'s `omni_engine` already routes per-dataset between a 2D and 3D
  student/teacher pair via `is_3d_list`/`_route`. Calling it with an **all-3D**
  `dataset_list` degrades cleanly to a 3D-only run — it still constructs an idle,
  zero-head 2D Swin model alongside (wastes some VRAM, never trained or evaluated
  since no 2D dataset ever routes to it). Not worth guarding out: the checkpoint
  resume / crash-proof-resume paths reference `model`/`teacher` unconditionally in
  several places, and this GPU's crash history makes that code worth not touching
  unless the waste actually causes an OOM.
- **Correction found while re-checking before writing the plan:** the 2D downstream
  fine-tuning driver was only ever *planned*
  (`docs/superpowers/plans/2026-08-12-downstream-finetune.md`, committed `7a92db1`)
  — `finetune_downstream.py` was never created, and `trainer.train_downstream_epoch`
  / `models.load_finetune_backbone` don't exist in either the committed history or
  the current uncommitted diff. That plan's Task 1/Task 2 code (a plain supervised
  train-epoch function and a heads-stripping checkpoint loader) is dataset-agnostic
  by design — nothing in it is 2D-specific — so it gets implemented directly here,
  driving `ArkR3D`/3D data instead of `ArkSwinTransformer`/2D data, rather than
  porting from a 2D file that doesn't exist. The 2D-specific parts of that old plan
  (`finetune_downstream.py` itself, argparse choices over `MEDMNIST_2D_KEYS`) are not
  built — out of scope here, since the target benchmarks are all 3D.

Also found: the `ark_medmnist` conda env has had every package uninstalled
(`conda-meta/history` shows a full removal, `conda-meta` itself is empty) despite
training logs proving it worked as of Aug 15 23:23. `environment.yml` (committed,
Task 1 of the 2D plan) is the recreation spec. This blocks everything and is fixed
first.

## Scope

Four phases, executed in order:

**Phase 0 — Unblock:** recreate `ark_medmnist` from `environment.yml`; run every
existing test plus the three new uncommitted test files
(`test_ema_buffer.py`, `test_medmnist3d_dataloader.py`, `test_models_medmnist3d.py`)
to confirm the already-written 3D infrastructure actually works; commit the
uncommitted WIP (it's foundational, dataset-agnostic, needed regardless of this
plan) as logical units.

**Phase 1 — Pretrain:** `pretrain_medmnist3d_run.py`, a new driver paralleling
`pretrain_medmnist2d_run.py`, calling the existing `omni_engine` with
`dataset_list = list(DATASET_MAP_3D)` (all 6 keys).

**Phase 2 — Fine-tune:** `finetune_downstream_3d.py`, a new driver paralleling
`finetune_downstream.py`, run once per 3D dataset against the pretrained checkpoint.

**Phase 3 — Tune loop:** not new code — an execution procedure (see below) run
directly: pretrain once, fine-tune all 6, retry under-target datasets with adjusted
hyperparameters up to a bounded number of attempts, report final results table.

## What's new

### `pretrain_medmnist3d_run.py`

Same shape as `pretrain_medmnist2d_run.py`: builds an `args` namespace and calls
`omni_engine(args, model_path, output_path, dataset_list, datasets_config, ...)`
directly (no `main_ark.py` 40-option parser — same rationale already established
for `finetune_downstream.py`: only a narrow slice of fields actually matters).

Key args:
- `dataset_list = list(DATASET_MAP_3D)` — all 6: OrganMNIST3D, NoduleMNIST3D,
  AdrenalMNIST3D, FractureMNIST3D, VesselMNIST3D, SynapseMNIST3D.
- `datasets_config = DATASETS_CONFIG_3D` (already built in `medmnist3d_dataloader.py`).
- `model_name` irrelevant to the actual encoder used here (0 2D datasets means the
  2D branch is never trained/evaluated) but still required by `build_omni_model`'s
  signature for the idle 2D model — reuse `"swin_tiny"` for consistency with the 2D
  run rather than inventing a new value.
- `batch_size_3d=16`, `lr_3d=1e-3`, `momentum_3d=0.9` — carried over from the
  abandoned 18-combined run, where these were already tuned for this GPU's memory
  ceiling. `# ponytail: reusing prior-validated defaults rather than re-deriving`.
  `batch_size` (2D, unused) set to something harmless like 32.
  `pretrained_weights_3d`: Kinetics init, applied the same two-step way
  `pretrain_medmnist18_run.py` already does (`build_omni_model_3d` + probe +
  `load_kinetics_backbone` + save + point `args.pretrained_weights_3d` at it).
- `pretrain_epochs=20` (matches the 2D run's precedent; adjustable in Phase 3 if
  results are short of target).
- `use_amp=True`, `crash_proof_resume=True` — this GPU has a real crash history;
  keep the defensive machinery on even though the 3D-only path is far simpler than
  the dual-encoder 18-combined one that actually crashed.
- `exp_name="medmnist3d_r3d18"`.

Output: `Models/swin_tiny_medmnist3d_r3d18/Ark_Plus_6ds_<hash>/medmnist3d_r3d18/...`
checkpoints (same layout `omni_engine` already produces), periodic per-dataset
test-AUC reports in `train.log`/`epoch_metrics.jsonl` (already built by the
uncommitted `engine.py` changes).

### `trainer.train_downstream_epoch` / `models.load_finetune_backbone`

Implemented per the already-reviewed design in
`docs/superpowers/plans/2026-08-12-downstream-finetune.md` Tasks 1-2, verbatim
(both are dataset/model-agnostic: plain supervised train-epoch step, and a
checkpoint loader that strips `omni_heads.*` keys before `load_state_dict`).

### `finetune_downstream_3d.py`

New driver, following the same design as that plan's (unbuilt) `finetune_downstream.py`
Task 3, adapted for 3D:
- `--dataset` choices = the 6 `DATASET_MAP_3D` keys instead of `MEDMNIST_2D_KEYS`.
- Data: `MedMNIST3DWrapper(dataset, split=...)` directly (no `build_transform_classification`
  — 3D augmentation lives in `Augment3D`, already wired into the wrapper).
- Model: `build_omni_model_3d(model_args, [num_classes])`, `load_finetune_backbone(model,
  checkpoint, key="teacher_3d")` (the 3D-only pretraining checkpoint's teacher weights
  live under `teacher_3d`, not `teacher` — that key holds the idle unused 2D model's
  state and must not be loaded here).
- `test_classification(model, 0, test_loader, device, multiclass, is_3d=True)`.
- Same early-stop / best-checkpoint-by-val-loss / final-test-AUC structure as the 2D
  driver, same output layout (`Models/finetune_<dataset>/`, `Outputs/finetune_<dataset>/`).
- Adds a `BENCHMARKS_3D` dict (the table above) and, after computing `overall_auc`,
  prints one of `PASS`, `WITHIN_1PCT`, or `FAIL` against the target (datasets with no
  target print `NO_TARGET`). This comparison is reporting only — it does not change
  training behavior.

### Environment repair

`conda env create -f environment.yml` (or `update --prune` if the env shell still
exists but is empty, whichever conda accepts cleanly) to restore
python/torch/torchvision/timm/medmnist/etc. No changes to `environment.yml` itself
expected — it was already correct as of the last successful run.

## Autonomous tuning procedure (Phase 3, executed directly, not scripted)

1. Run `pretrain_medmnist3d_run.py` to completion (or until crash-proof-resume
   carries it through to `pretrain_epochs`).
2. Run `finetune_downstream_3d.py --dataset <key>` for all 6 datasets against the
   final pretrained checkpoint.
3. For any dataset that misses its target by more than 1%: retry with one
   hyperparameter change per attempt, in order, up to 3 attempts total per dataset:
   lower `--lr` (1e-3 → 3e-4 → 1e-4), then more `--epochs` (20 → 40) if lr changes
   alone don't close the gap.
4. If, after step 3, a dataset (or several) still misses its target: extend
   pretraining by additional epochs from the last checkpoint (crash-proof-resume
   already supports this) once, then re-run step 2-3 for the still-short datasets
   only.
5. Report a final table: dataset, target, achieved AUC, PASS/WITHIN_1PCT/FAIL,
   attempts taken.

This is a bounded procedure (not an open-ended search) — at most one pretraining
extension round and 3 fine-tune attempts per dataset before reporting results as-is.

## Error handling / operational notes

- `crash_proof_resume=True` on the pretraining run given this GPU's documented
  instability; each fine-tuning run has no resume (per the existing 2D driver's
  design: a single-dataset fine-tune is cheap enough to just rerun from scratch).
- If the environment recreation itself fails (e.g. a package no longer resolves),
  stop and report rather than trying alternate package versions blind — the pinned
  `environment.yml` is the source of truth for "faithful to what was already
  validated working."

## Testing

- Run existing suite (`test_trainer_amp.py`, `test_models_medmnist.py`,
  `test_engine_resume_indexing.py`, etc.) plus the three new uncommitted 3D test
  files, after the environment is restored — confirms Phase 0's WIP is solid before
  building on it.
- `pretrain_medmnist3d_run.py`: no new unit tests (it's a thin args-assembly driver
  like `pretrain_medmnist2d_run.py`, which also has none) — validated by a real
  short smoke run (1-2 epochs) before committing to the full `pretrain_epochs=20`
  run.
- `finetune_downstream_3d.py`: unit tests for its pure helpers only
  (`_early_stop`, `parse_args`, benchmark-comparison logic), same pattern as
  `test_finetune_downstream.py`. Validated end-to-end with a real smoke run on the
  smallest 3D dataset before trusting it for all 6.

## Out of scope

- Modifying or resuming the abandoned 18-combined joint run or its logs.
- Any change to 2D-only behavior or the 12-dataset pipeline.
- Guarding `omni_engine` against building the idle 2D model when `dataset_list` is
  all-3D (flagged as a `ponytail:` deferred item above, not built).
- An open-ended/generic hyperparameter search framework — the tuning procedure
  above is a fixed, bounded sequence, not a new configurable sweep tool.
