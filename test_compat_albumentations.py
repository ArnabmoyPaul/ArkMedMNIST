#!/usr/bin/env python3
"""Self-check for _compat_albumentations.py compatibility shim.

This test verifies that the patched albumentations classes
(RandomBrightness, RandomContrast, JpegCompression, IAAAdditiveGaussianNoise)
can be instantiated successfully.

Run with: python test_compat_albumentations.py
"""

if __name__ == "__main__":
    # Import the shim FIRST (triggers patching before albumentations usage)
    import _compat_albumentations

    # Now import albumentations - the patched classes should be available
    import albumentations as A

    print("Testing _compat_albumentations.py patches...\n")

    # Test 1: RandomBrightness (patched alias)
    print("1. Testing RandomBrightness(limit=0.2, p=0.5)...")
    try:
        obj = A.RandomBrightness(limit=0.2, p=0.5)
        print(f"   ✓ Created: {type(obj).__name__}")
        assert hasattr(obj, '__call__'), "RandomBrightness should be callable"
        print("   ✓ Object is callable")
    except Exception as e:
        print(f"   ✗ FAILED: {e}")
        raise

    # Test 2: RandomContrast (patched alias)
    print("\n2. Testing RandomContrast(limit=0.2, p=0.5)...")
    try:
        obj = A.RandomContrast(limit=0.2, p=0.5)
        print(f"   ✓ Created: {type(obj).__name__}")
        assert hasattr(obj, '__call__'), "RandomContrast should be callable"
        print("   ✓ Object is callable")
    except Exception as e:
        print(f"   ✗ FAILED: {e}")
        raise

    # Test 3: JpegCompression (alias to ImageCompression)
    print("\n3. Testing JpegCompression() with defaults...")
    try:
        obj = A.JpegCompression()
        print(f"   ✓ Created: {type(obj).__name__}")
        assert hasattr(obj, '__call__'), "JpegCompression should be callable"
        print("   ✓ Object is callable")
    except Exception as e:
        print(f"   ✗ FAILED: {e}")
        raise

    # Test 4: IAAAdditiveGaussianNoise (alias to GaussNoise)
    print("\n4. Testing IAAAdditiveGaussianNoise() with defaults...")
    try:
        obj = A.IAAAdditiveGaussianNoise()
        print(f"   ✓ Created: {type(obj).__name__}")
        assert hasattr(obj, '__call__'), "IAAAdditiveGaussianNoise should be callable"
        print("   ✓ Object is callable")
    except Exception as e:
        print(f"   ✗ FAILED: {e}")
        raise

    print("\n" + "="*60)
    print("All _compat_albumentations patches verified successfully!")
    print("="*60)
