"""Compatibility shim for albumentations >= 1.0.0.

CRITICAL: This module MUST be imported before any `from dataloader import ...`
or `import dataloader` statements anywhere in the codebase. The import order
matters because this module patches the albumentations namespace as a side effect.

Patches these albumentations names that were removed in v1.0.0+:
  - RandomBrightness → wrapper around RandomBrightnessContrast
  - RandomContrast → wrapper around RandomBrightnessContrast
  - JpegCompression → alias to ImageCompression
  - IAAAdditiveGaussianNoise → alias to GaussNoise

The Ark_Plus code was written for older albumentations API (v0.5.x); this
shim maintains compatibility with albumentations==1.4.24 by providing the
missing/renamed classes that dataloader.py imports at module level.

Without this shim imported FIRST, any code that imports dataloader.py will
fail with ImportError on the missing classes.

Example (correct):
    import _compat_albumentations  # Import shim first
    from dataloader import ChestXray14

Example (wrong):
    from dataloader import ChestXray14  # Will fail with ImportError
"""
import albumentations as A

# ponytail: version 1.4.24 removed several old IAA/deprecated transforms
# These mappings provide the closest modern equivalents

# RandomBrightness: removed in 1.0.0, replace with RandomBrightnessContrast(contrast_limit=0)
if not hasattr(A, 'RandomBrightness'):
    class RandomBrightness(A.RandomBrightnessContrast):
        """Compatibility wrapper: RandomBrightness adjusts only brightness."""
        def __init__(self, limit=0.2, p=0.5):
            super().__init__(brightness_limit=limit, contrast_limit=0, p=p)
    A.RandomBrightness = RandomBrightness

# RandomContrast: removed in 1.0.0, replace with RandomBrightnessContrast(brightness_limit=0)
if not hasattr(A, 'RandomContrast'):
    class RandomContrast(A.RandomBrightnessContrast):
        """Compatibility wrapper: RandomContrast adjusts only contrast."""
        def __init__(self, limit=0.2, p=0.5):
            super().__init__(brightness_limit=0, contrast_limit=limit, p=p)
    A.RandomContrast = RandomContrast

# JpegCompression: renamed to ImageCompression in 1.4.x
if not hasattr(A, 'JpegCompression') and hasattr(A, 'ImageCompression'):
    A.JpegCompression = A.ImageCompression

# IAAAdditiveGaussianNoise: removed, use GaussNoise as closest equivalent
if not hasattr(A, 'IAAAdditiveGaussianNoise') and hasattr(A, 'GaussNoise'):
    A.IAAAdditiveGaussianNoise = A.GaussNoise
