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

**Albumentations API break.** `dataloader.py` imports `RandomBrightness`, `RandomContrast`,
`JpegCompression`, and `IAAAdditiveGaussianNoise` from `albumentations`, all removed or
renamed since the repo's era (now on `albumentations==1.4.24`, pinned in `environment.yml`
for the same torch/Python compatibility reason as timm above). `_compat_albumentations.py`
patches the four names back onto the `albumentations` module as a side effect of import;
it must be imported before any `dataloader` import anywhere in the codebase (`verify_env.py`
and `main_ark.py` both do this first). A compatibility shim for a library version gap, not
a change to Ark+'s augmentation design.

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
