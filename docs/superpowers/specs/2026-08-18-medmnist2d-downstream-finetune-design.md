# 2D Downstream Fine-Tuning: Closing the 18-Dataset Benchmark Gap

## Goal

Fine-tune and evaluate the completed 12-dataset 2D omni-pretrained encoder
(`Models/swin_tiny_medmnist2d_swintiny/.../medmnist2d_swintiny/..._chestmnist24.pth.tar`,
epoch 24, the final checkpoint of that run) separately on each of the 12
MedMNIST-2D datasets, matching the Ark+ paper's downstream-eval protocol —
the same procedure already completed for the 6 3D datasets
(`finetune_downstream_3d.py`). Once done, report one combined 18-dataset
table (target AUC, achieved AUC, PASS/WITHIN_1PCT/FAIL) against the
professor-supplied targets ("individual training (AUC), reported by the
original authors"):

| Dataset | Target AUC |
|---|---|
| chestmnist | 0.773 |
| pathmnist | 0.989 |
| dermamnist | 0.912 |
| octmnist | 0.958 |
| pneumoniamnist | 0.962 |
| retinamnist | 0.716 |
| breastmnist | 0.866 |
| bloodmnist | 0.997 |
| tissuemnist | 0.932 |
| organamnist | 0.998 |
| organcmnist | 0.993 |
| organsmnist | 0.975 |

(The 3D targets in this same table — OrganMNIST3D 0.994, VesselMNIST3D
0.905, AdrenalMNIST3D 0.828, FractureMNIST3D 0.725, NoduleMNIST3D 0.875,
SynapseMNIST3D 0.851 — already match `finetune_downstream_3d.py`'s
`BENCHMARKS_3D` exactly; no change needed there.)

Success per dataset: meet or beat the target, or land within 1% of it —
identical criterion to the 3D work.

## Context

The 12-2D omni-pretraining run (`pretrain_medmnist2d_run.py`) completed all
25 epochs (0-24) on 2026-08-13 — confirmed via `epoch_metrics.jsonl`'s final
entry (epoch 24 teacher/student mAUC per dataset). A downstream fine-tuning
driver for this checkpoint was designed and approved on 2026-08-12
(`docs/superpowers/specs/2026-08-12-downstream-finetune-design.md`,
committed `7a92db1`) but **never implemented** — `finetune_downstream.py`
does not exist in this repo, uncommitted or otherwise. The 3D work that
followed (`docs/superpowers/specs/2026-08-16-medmnist3d-ark-plus-replication-design.md`)
explicitly built `trainer.train_downstream_epoch` and
`models.load_finetune_backbone` per that same 2026-08-12 plan's Tasks 1-2
(both dataset/modality-agnostic) but left the 2D driver itself (Task 3, and
the file `finetune_downstream.py`) out of scope, since the 3D targets were
the only ones available at the time.

Separately: a joint single-loop 18-dataset run (`pretrain_medmnist18_run.py`,
cycling all 12 2D + 6 3D datasets through one `omni_engine` call per epoch)
was attempted twice (Aug 11-15) and abandoned after repeated crashes — the
logs cut off mid-progress-bar both times, never completing even one full
epoch across all 18 datasets. This is confirmed **not** what the original
Ark+ paper does (2D and 3D are trained as two separate pipelines there), and
is explicitly out of scope for this spec — this plan produces the 18-dataset
benchmark table from the two independently-pretrained encoders (2D-12,
3D-6), not from a joint run.

## Scope

One new file, `finetune_downstream.py`, run once per 2D dataset:

```
python finetune_downstream.py --dataset chestmnist
```

Structurally identical to `finetune_downstream_3d.py` — same
`parse_args`/`_early_stop`/`_compare_to_benchmark` shape, same output
layout (`Models/finetune_<dataset>/`, `Outputs/finetune_<dataset>/`),
same PASS/WITHIN_1PCT/FAIL reporting — swapped from 3D to 2D data/model
plumbing. Each invocation is a fully independent run (own model, own
checkpoint, own log, own final test-AUC number); no crash-proof resume
(a single-dataset fine-tune is cheap enough to just rerun from scratch,
same reasoning as the existing 3D driver and the original approved 2D
spec).

## What gets reused as-is

- `build_omni_model(args, [num_classes])` (`models.py`) — same function
  `pretrain_medmnist2d_run.py` used, called with a 1-element class list.
- `dict_dataloarder[dataset](...)` + `build_transform_classification(...)`
  (`dataloader.py`, `medmnist_dataloader.py`) — identical per-dataset
  loader construction already used in `pretrain_medmnist2d_run.py`, for
  one dataset instead of a loop of 12.
- `trainer.train_downstream_epoch()`, `trainer.evaluate()`,
  `trainer.test_classification()` — already built (2026-08-16 plan,
  dataset-agnostic), currently only driven by the 3D file. Reused
  unchanged, called with 2D data.
- `models.load_finetune_backbone(model, checkpoint, key="teacher")` — the
  2D-only pretraining run never trained a 3D encoder (`has_3d=False` in
  `omni_engine`, confirmed by reading `engine.py`'s `checkpoint_now`),
  so its checkpoint only ever has a `teacher` key, not `teacher_3d` —
  the function's existing default (`key='teacher'`) already fits with no
  changes.
- `metric_AUROC` (`utils.py`), `_early_stop`/`_compare_to_benchmark`
  logic — copied verbatim from `finetune_downstream_3d.py` (same
  tolerance: within 0.01 AUC of target = `WITHIN_1PCT`).
- `MEDMNIST_2D_KEYS` (`medmnist_dataloader.py`) for `--dataset` choices,
  `get_config('datasets_config_medmnist.yaml')` for per-dataset
  `diseases`/`task_type` (identical config source `pretrain_medmnist2d_run.py`
  already uses).

## What's new

**`finetune_downstream.py`** (new driver script), differences from
`finetune_downstream_3d.py`:
- `--dataset` choices = `MEDMNIST_2D_KEYS` (12 keys) instead of
  `DATASET_MAP_3D`.
- `--checkpoint` default:
  `Models/swin_tiny_medmnist2d_swintiny/Ark_Plus_pathmnist_bloodmnist_dermamnist_octmnist_pneumoniamnist_retinamnist_breastmnist_tissuemnist_organamnist_organcmnist_organsmnist_chestmnist/medmnist2d_swintiny/Ark_Plus_pathmnist_bloodmnist_dermamnist_octmnist_pneumoniamnist_retinamnist_breastmnist_tissuemnist_organamnist_organcmnist_organsmnist_chestmnist24.pth.tar`
  (epoch 24, the final checkpoint of the completed run — no "best epoch"
  selection logic, consistent with `DEVIATIONS.md`'s explicit note that
  best-checkpoint-by-mAUC selection doesn't exist in the real `omni_engine`
  and is deferred; the 3D work followed the same rule, always fine-tuning
  from whatever the latest completed pretrain epoch was).
- `--batch_size` default `32` (matches the 2D pretrain run's batch size;
  3D used 16 for heavier volumetric batches — not applicable here).
- Data loading uses `dict_dataloarder[args.dataset](...)` +
  `build_transform_classification(...)` (three calls: train/val/test,
  identical shape to `pretrain_medmnist2d_run.py`'s loop body for one
  dataset) instead of `MedMNIST3DWrapper`.
- Model: `build_omni_model(model_args, [len(diseases)])` instead of
  `build_omni_model_3d`.
- `test_classification(model, 0, test_loader, device, multiclass)` —
  `is_3d` left at its default (`False`), instead of passing `True`.
- `BENCHMARKS_2D` dict (the table above) instead of `BENCHMARKS_3D`.

No changes to any existing file.

## Execution plan (run directly, not scripted)

1. Run `finetune_downstream.py --dataset <key>` for all 12 keys,
   sequentially in one background job (single GPU — no parallelization),
   each writing its own log (`finetune_<dataset>.log`) and
   `Outputs/finetune_<dataset>/results.json`.
2. If a run fails partway (e.g. OOM on `chestmnist`, the largest dataset
   at ~78K train images), only that one dataset needs a rerun — no shared
   state to corrupt, matching the existing 3D fine-tune runs' behavior.
3. Once all 12 finish, compile one combined 18-dataset markdown table
   (target, achieved, verdict) by hand from the 18 `results.json` files —
   no new report-generation script, matching how the 3D table was
   assembled and reported in this session.

## Error handling / operational notes

- No crash-proof resume for fine-tuning (per the original 2026-08-12 spec's
  reasoning, unchanged): a single-dataset fine-tune is cheap enough that
  redoing it from scratch is fine.
- `chestmnist` is multi-label classification (`BCEWithLogitsLoss`) with 14
  disease classes — largest and slowest of the 12; the rest are
  multi-class (`CrossEntropyLoss`). Both branches already exist in
  `finetune_downstream_3d.py`'s criterion-selection line, reused as-is.

## Testing

- Unit tests for the pure helpers only (`_early_stop`, `parse_args`,
  `_compare_to_benchmark`), in a new `test_finetune_downstream.py`,
  mirroring `test_finetune_downstream_3d.py` exactly (same test names,
  `BENCHMARKS_2D` spot-check instead of the Synapse-target check).
- Validated end-to-end with a real smoke run on the smallest 2D dataset
  (`breastmnist`, 546 train samples) before trusting it for all 12.

## Out of scope

- Retrying or resuming the abandoned joint 18-dataset single-loop run
  (`pretrain_medmnist18_run.py`) or its logs — confirmed not the paper's
  actual design; not touched by this spec.
- Any change to the 3D fine-tuning driver, its benchmark table, or its
  checkpoints.
- A new automated 18-dataset report generator — the combined table is
  compiled by hand from the 18 independent `results.json` files, once,
  after all runs finish.
- Re-tuning the 2D pretraining run itself (more epochs, different LR,
  etc.) — this spec only covers fine-tuning the already-completed
  checkpoint. If 2D datasets come up short after fine-tuning, a tuning
  procedure analogous to the 3D plan's Phase 3 would be a follow-up, not
  part of this spec.
