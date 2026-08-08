# Environment Setup & CUDA Verification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Pin a reproducible conda environment for the Ark+ MedMNIST replication and prove CUDA/GPU visibility before any training code runs.

**Architecture:** One `environment.yml` capturing the exact package set already verified working in the `ark_medmnist` conda env on this machine, plus a standalone `verify_env.py` that prints torch/CUDA/GPU/VRAM info and exits non-zero if CUDA is unavailable.

**Tech Stack:** conda, PyTorch 2.13 (cu130), timm, medmnist, scikit-learn.

## Global Constraints

- Deliverable 1 from the Ark+ replication brief: `environment.yml` pinned and reproducible, must include a CUDA-matched PyTorch, `timm`, `medmnist`, `scikit-learn`, `numpy`, `pillow`, `tqdm`.
- Verification script must print `torch.__version__`, `torch.cuda.is_available()`, the GPU name, and VRAM total, and must fail loudly (non-zero exit) if CUDA is missing.
- GPU is an RTX 4060 with 8188 MiB VRAM (confirmed via `nvidia-smi`). Do not assume 16 GB.
- The `ark_medmnist` conda env already exists on this machine with working versions of every required package (confirmed via `conda list -n ark_medmnist`) — pin `environment.yml` to those exact versions rather than guessing newer ones.

---

### Task 1: Write `environment.yml`

**Files:**
- Create: `environment.yml`

**Interfaces:**
- Consumes: nothing.
- Produces: a conda env spec other tasks/scripts assume is active (`ark_medmnist`).

- [ ] **Step 1: Write the pinned environment file**

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
  - pip
  - pip:
      - torch==2.13.0+cu130
      - torchvision==0.28.0+cu130
      - --extra-index-url https://download.pytorch.org/whl/cu130
      - timm==1.0.28
      - medmnist==3.0.2
      - tqdm==4.68.4
```

`pip`'s `--extra-index-url` line must come before the packages that need it in the same `pip:` block for conda's pip sub-installer to pick it up; conda-lock style tools sort this differently, but plain `conda env create` installs the `pip:` list in file order with `pip install -r <generated-file>`, so keep the index URL line directly above `timm`/`medmnist`/`tqdm` are pure-PyPI and don't need it — only `torch`/`torchvision` do. Reorder so the extra-index-url line sits immediately after `- pip` and before the two CUDA wheels:

```yaml
  - pip:
      - --extra-index-url https://download.pytorch.org/whl/cu130
      - torch==2.13.0+cu130
      - torchvision==0.28.0+cu130
      - timm==1.0.28
      - medmnist==3.0.2
      - tqdm==4.68.4
```

Use this corrected ordering in the file.

- [ ] **Step 2: Commit**

```bash
git add environment.yml
git commit -m "add pinned environment.yml for ark_medmnist conda env"
```

---

### Task 2: Write and run the CUDA/VRAM verification script

**Files:**
- Create: `verify_env.py`

**Interfaces:**
- Consumes: the active conda environment's installed packages.
- Produces: nothing importable — this is a standalone CLI check other tasks/humans run manually before training (`python verify_env.py`).

- [ ] **Step 1: Write the verification script**

```python
"""Verify the training environment before running anything expensive.

Run: python verify_env.py
Exits non-zero if CUDA is unavailable or a required package is missing/mismatched.
"""
import sys

REQUIRED = ["torch", "torchvision", "timm", "medmnist", "sklearn", "numpy", "PIL", "tqdm"]


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

    print(f"torch.__version__        = {torch.__version__}")
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
              f"this is the 8 GB 4060 tier, not the 16 GB 4060 Ti. "
              f"Batch sizes must be sized accordingly.")

    # Smoke-test an actual CUDA allocation + kernel launch, not just is_available().
    x = torch.randn(1024, 1024, device="cuda")
    y = x @ x
    torch.cuda.synchronize()
    print(f"CUDA matmul smoke test     = OK (result sum={y.sum().item():.2f})")


def main():
    check_imports()
    check_cuda()
    print("\nEnvironment verification PASSED.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it and confirm it passes**

Run: `conda run -n ark_medmnist python verify_env.py`

Expected output ends with:
```
GPU name                  = NVIDIA GeForce RTX 4060
GPU total VRAM             = 7.99 GiB (...)
CUDA matmul smoke test     = OK (result sum=...)

Environment verification PASSED.
```

If VRAM shown is well below ~7.5 GiB free at the *start* of a training run (check with `nvidia-smi --query-gpu=memory.free --format=csv` first), close other GPU-heavy applications (games, browsers with hardware acceleration, other conda envs holding CUDA contexts) before launching training — `verify_env.py` reports total VRAM, not currently-free VRAM, so it will pass even when another process is hogging memory.

- [ ] **Step 3: Commit**

```bash
git add verify_env.py
git commit -m "add CUDA/VRAM verification script"
```

---

## Self-Review Notes

- Spec coverage: `environment.yml` (pinned, CUDA-matched torch + timm/medmnist/scikit-learn/numpy/pillow/tqdm) — Task 1. Verification script printing torch version, CUDA availability, GPU name, VRAM total, failing loudly without CUDA — Task 2. Both deliverable-1 requirements covered.
- No placeholders: both files are complete, runnable code.
- Known gap (accepted): `verify_env.py` checks *total* VRAM, not *free* VRAM, because "training requires 8 GB free" is a runtime precondition, not an install-time one — noted explicitly in Task 2's step so it isn't silently missed.
