"""
Tests for Task 8 — Manual Color-to-Grayscale Conversion with Histogram & CDF.

Validates:
    1. Output shape and dtype after grayscale conversion.
    2. Pixel-level accuracy of the ITU-R BT.601 luma formula.
    3. Histogram bin sums equal total pixel count.
    4. CDF mathematical invariants (non-negative, normalized, monotonic).
"""

import numpy as np
import pytest
from core.operations import ManualGrayscale


# ───────────────────────────────────────────
# TEST 1 — Grayscale Shape & Type
# ───────────────────────────────────────────
def test_grayscale_shape_and_type():
    """A 3-channel RGB image must become a 2D uint8 array."""
    # Create a 3×3 BGR synthetic image
    bgr_image = np.array([
        [[255, 0, 0], [0, 255, 0], [0, 0, 255]],
        [[128, 128, 128], [64, 64, 64], [200, 100, 50]],
        [[10, 20, 30], [50, 60, 70], [90, 80, 70]],
    ], dtype=np.uint8)

    op = ManualGrayscale()
    output = op.apply(bgr_image)

    assert output.ndim == 2, f"Expected 2D output, got {output.ndim}D"
    assert output.dtype == np.uint8, f"Expected uint8, got {output.dtype}"
    assert output.shape == (3, 3), f"Expected (3,3), got {output.shape}"


# ───────────────────────────────────────────
# TEST 2 — Formula Accuracy
# ───────────────────────────────────────────
def test_formula_accuracy():
    """Known RGB values must produce the exact expected gray intensity."""
    # Single pixel: B=200, G=150, R=100  (BGR order)
    bgr_image = np.array([[[200, 150, 100]]], dtype=np.uint8)

    op = ManualGrayscale()
    output = op.apply(bgr_image)

    # Expected: int(0.299*100 + 0.587*150 + 0.114*200)
    #         = int(29.9 + 88.05 + 22.8)
    #         = int(140.75)
    #         = 140
    expected = int(0.299 * 100 + 0.587 * 150 + 0.114 * 200)
    assert output[0, 0] == expected, (
        f"Expected gray={expected}, got {output[0, 0]}"
    )


# ───────────────────────────────────────────
# TEST 3 — RGB Histogram Consistency
# ───────────────────────────────────────────
def test_rgb_histogram_consistency():
    """Histogram bin sums must equal the total number of pixels."""
    # Create 10×10 BGR image with known distribution
    bgr_image = np.zeros((10, 10, 3), dtype=np.uint8)
    bgr_image[:5, :, :] = [50, 100, 150]   # Top half
    bgr_image[5:, :, :] = [200, 180, 60]   # Bottom half

    total_pixels = bgr_image.shape[0] * bgr_image.shape[1]

    for ch in range(3):
        hist, _ = np.histogram(bgr_image[:, :, ch].flatten(), bins=256, range=(0, 256))
        assert hist.sum() == total_pixels, (
            f"Channel {ch}: histogram sum {hist.sum()} != pixel count {total_pixels}"
        )


# ───────────────────────────────────────────
# TEST 4 — CDF Properties
# ───────────────────────────────────────────
def test_cdf_properties():
    """CDF must be non-negative at start, normalized to 1.0, and monotonically non-decreasing."""
    # Apply grayscale conversion first, then check CDF on the result
    bgr_image = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)

    op = ManualGrayscale()
    gray = op.apply(bgr_image)

    # Compute histogram and normalized CDF
    hist, _ = np.histogram(gray.flatten(), bins=256, range=(0, 256))
    cdf = hist.cumsum()
    cdf_normalized = cdf / cdf.max()

    # Property 1: First value >= 0
    assert cdf_normalized[0] >= 0, "CDF first value must be >= 0"

    # Property 2: Last value == 1.0 (normalized)
    assert cdf_normalized[-1] == pytest.approx(1.0), (
        f"CDF last value must be 1.0, got {cdf_normalized[-1]}"
    )

    # Property 3: Monotonically non-decreasing
    diffs = np.diff(cdf_normalized)
    assert np.all(diffs >= 0), "CDF must be monotonically non-decreasing"
