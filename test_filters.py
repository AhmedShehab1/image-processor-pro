from filters import compute_histogram, compute_cdf
import numpy as np

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

test_histogram_logic()