# Downstream fine-tuning from the epoch-20 checkpoint

## Goal

Take the omni-pretrained encoder (epoch 20 of the 12-2D-MedMNIST run,
`Models/swin_tiny_medmnist2d_swintiny/.../medmnist2d_swintiny/..._chestmnist20.pth.tar`,
teacher weights — student mAUC peaked here at 0.8982, teacher at 0.9100) and
measure downstream transfer: fine-tune and evaluate it separately on each of
the 12 MedMNIST-2D datasets, one dataset per task/head, matching the Ark+
paper's own downstream-eval protocol.

## Scope

One new script, `finetune_downstream.py`, run once per dataset:

```
python finetune_downstream.py --dataset chestmnist
```

Each invocation is a fully independent run: its own model, own checkpoint,
own log, own final test-AUC number. No cross-dataset state. If a run gets
killed mid-training, only that dataset needs to be rerun — no crash-proof
resume machinery, unlike the joint 25-epoch pretraining run (Task 7), because
a single-dataset fine-tune is short enough that redoing it from scratch is
cheap.

## What gets reused as-is

- `build_omni_model(args, [num_classes])` (`models.py`) — same function the
  pretraining run used, called with a 1-element class list instead of 12.
- `dict_dataloarder[dataset](...)` + `build_transform_classification(...)`
  (`dataloader.py`, `medmnist_dataloader.py`) — identical per-dataset loader
  construction already used in `pretrain_medmnist2d_run.py`, for one dataset
  instead of a loop of 12.
- `trainer.evaluate()` and `trainer.test_classification()` (`trainer.py`) —
  already single-head (`use_head_n` param), single-dataset, no dependency on
  the self-distillation teacher. Used unchanged for val-loss tracking and
  final test AUC.
- `metric_AUROC` (`utils.py`) for the final AUC report, same as `engine.py`'s
  test block.
- Criterion selection (`CrossEntropyLoss` vs `BCEWithLogitsLoss` keyed off
  `task_type` in `datasets_config_medmnist.yaml`) — same one-line branch
  already in `engine.py`.

## What's new

**`trainer.train_downstream_epoch()`** (new function, next to the existing
train/eval helpers in `trainer.py`): a plain supervised train loop — forward,
loss, backward, step — reusing `MetricLogger`/`ProgressLogger` for logging
consistency with the rest of the file. `trainer.train_one_epoch()` is not
reused here because it's hard-wired to the omni self-distillation loss
(teacher forward pass + EMA update + MSE consistency term against the
momentum schedule); none of that applies to plain downstream fine-tuning.
Dataset `__getitem__` still yields `(view1, view2, target)` triples (the
dataset class is shared, unchanged) — the new loop ignores `view2`, same as
`evaluate()` already does.

**`finetune_downstream.py`** (new driver script):
- Args: `--dataset` (required, one of the 12 MedMNIST-2D keys), `--checkpoint`
  (defaults to the epoch-20 path above), `--epochs` (default 20), `--lr`
  (default 1e-3), `--patience` (default 5).
- Loads the checkpoint's `teacher` state dict into a fresh single-head model,
  stripping `omni_heads.*` keys before `load_state_dict(strict=False)` — the
  checkpoint's 12 heads are keyed by position (`omni_heads.0` through
  `omni_heads.11`) and don't correspond to the new single-head model's
  `omni_heads.0`, so loading them unfiltered would silently splice in the
  wrong dataset's head whenever class counts happen to match. This mirrors
  the existing `reinit_heads` filter in `engine.py`'s resume path.
- Same optimizer family as pretraining (`timm` momentum-SGD), same
  `crop_size=112` / `resize=128` / `batch_size=32` as the pretraining run for
  consistency.
- Trains up to `--epochs`, tracking val loss each epoch via
  `trainer.evaluate()`; stops early if val loss hasn't improved for
  `--patience` epochs. Saves the best-val-loss checkpoint (unlike the
  pretraining run, a single-dataset "best" is unambiguous, so this one does
  select and save a best checkpoint rather than saving unconditionally every
  epoch).
- At the end, runs `trainer.test_classification()` on the held-out test set
  using the best checkpoint's weights, reports AUC per class and overall
  mAUC via `metric_AUROC`, matching the reporting format already used in
  `engine.py`'s test block.
- Output layout: `Models/finetune_<dataset>/` for the checkpoint,
  `Outputs/finetune_<dataset>/` for the results text file — mirrors the
  existing `model_path` / `output_path` split in `engine.py`.

## Out of scope (not building)

- Looping all 12 datasets in one process — explicitly rejected in favor of
  12 independent invocations (see Scope).
- Crash-proof mid-run resume for fine-tuning — a single dataset's fine-tune
  is cheap enough to redo; the complexity Task 7 solved doesn't pay for
  itself here.
- Linear-probe mode — full fine-tune only, per the approved design choice.
- Held-out (non-MedMNIST) transfer datasets — scope is the 12 datasets
  already used in pretraining.
