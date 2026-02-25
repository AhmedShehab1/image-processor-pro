import numpy as np
from core.color_ops import compute_histogram, compute_cdf, normalize_image, equalize_image

def test_histogram_logic():
    # Create a 3x3 dummy image with specific values
    # Values: 0, 1, 2 (each appears 3 times)
    test_img = np.array([
        [0, 1, 2],
        [0, 1, 2],
        [0, 1, 2]
    ], dtype=np.uint8)

    # 1. Test Histogram
    hist = compute_histogram(test_img)
    
    # We expect indices 0, 1, and 2 to have a count of 3. Others should be 0.
    assert hist[0] == 3
    assert hist[1] == 3
    assert hist[2] == 3
    assert np.sum(hist) == 9  # Total pixels must match image size
    print("✅ Histogram Calculation: PASSED")

    # 2. Test CDF
    cdf = compute_cdf(hist)
    
    # Expectations for CDF (cumulative):
    # index 0: 3/9 = 0.33
    # index 1: (3+3)/9 = 0.66
    # index 2: (3+3+3)/9 = 1.0
    assert cdf[0] == 3/9
    assert cdf[1] == 6/9
    assert cdf[2] == 1.0
    assert cdf[255] == 1.0 # Should stay 1.0 until the end
    print("✅ CDF Calculation: PASSED")


def test_normalize_image_range():
    """
    Test 1: Normalization must stretch the image to the full 0-255 range.
    """
    # Create a low-contrast image (values only between 100 and 150)
    low_contrast = np.array([[100, 110], [140, 150]], dtype=np.uint8)
    
    normalized = normalize_image(low_contrast)
    
    # Mathematical Invariants:
    assert normalized.min() == 0   # The 100 should become 0
    assert normalized.max() == 255 # The 150 should become 255
    assert normalized.dtype == np.uint8
    print("✅ Normalization Range Test: Passed")

def test_normalize_solid_color():
    """
    Test 2: Normalization shouldn't crash on a solid color image (division by zero check).
    """
    solid = np.full((10, 10), 128, dtype=np.uint8)
    normalized = normalize_image(solid)
    
    # It should return the same image or at least not crash
    assert np.all(normalized == 128)
    print("✅ Normalization Solid Color Test: Passed")

def test_equalize_image_cdf_property():
    """
    Test 3: After equalization, the CDF should be approximately linear.
    """
    # Create a very "dark" image (most pixels at 10)
    dark_img = np.zeros((100, 100), dtype=np.uint8)
    dark_img[0:50, :] = 10
    dark_img[50:100, :] = 20
    
    equalized = equalize_image(dark_img)
    
    # Mathematical Invariant:
    # In an equalized image, the min value should be 0 and max should be 255 
    # because it spreads the distribution.
    assert equalized.min() == 0
    assert equalized.max() == 255
    
    # Check that the distribution changed
    hist_orig = np.histogram(dark_img, bins=256, range=(0,256))[0]
    hist_eq = np.histogram(equalized, bins=256, range=(0,256))[0]
    
    # The equalized histogram should have values spread out, not just at index 10 and 20
    assert np.count_nonzero(hist_eq) >= np.count_nonzero(hist_orig)
    print("✅ Equalization Distribution Test: Passed")

def test_idempotency():
    """
    Test 4: Equalizing an already perfectly equalized image shouldn't change it much.
    """
    # Create a gradient (already perfectly distributed)
    gradient = np.linspace(0, 255, 256).reshape(16, 16).astype(np.uint8)
    
    eq_once = equalize_image(gradient)
    eq_twice = equalize_image(eq_once)
    
    # After the first pass, the second pass should result in almost the same image
    # (Allowing for small rounding differences)
    assert np.allclose(eq_once, eq_twice, atol=1)
    print("✅ Idempotency Test: Passed")
