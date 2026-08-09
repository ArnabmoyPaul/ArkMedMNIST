"""Verify the training environment before running anything expensive.

Run: python verify_env.py
Exits non-zero if CUDA is unavailable or a required package is missing.
"""
import sys

# Load compatibility shims before other imports (RandomBrightness for albumentations >= 1.0)
import _compat_albumentations  # noqa: F401

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
