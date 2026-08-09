"""Compatibility shim for albumentations.

Provides missing/renamed classes for albumentations>=1.0.0.
The Ark_Plus code was written for older albumentations API; this shim
maintains compatibility with newer versions.
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
