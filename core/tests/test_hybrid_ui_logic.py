"""
Tests for Task 10 — Hybrid Image UI Logic.

Validates the standalone compute_hybrid() function that backs
both the pipeline operation and the real-time canvas preview:
    1. Output shape and dtype match Image A.
    2. Different sigma values produce meaningfully different results.
    3. Original input images are never mutated.
"""

import numpy as np
import pytest
from core.operations import compute_hybrid


# ───────────────────────────────────────────
# TEST 1 — Hybrid Output Shape & Dtype
# ───────────────────────────────────────────
def test_hybrid_output_shape():
    """Two 100×100 RGB images must produce a 100×100 RGB uint8 result."""
    img_a = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    img_b = np.random.randint(0, 256, (80, 120, 3), dtype=np.uint8)  # different size

    result = compute_hybrid(img_a, img_b, sigma_low=5.0, sigma_high=5.0)

    assert result.shape == (100, 100, 3), (
        f"Expected shape (100,100,3), got {result.shape}"
    )
    assert result.dtype == np.uint8, (
        f"Expected uint8, got {result.dtype}"
    )


# ───────────────────────────────────────────
# TEST 2 — Slider Sensitivity
# ───────────────────────────────────────────
def test_slider_sensitivity():
    """Different sigma_high values must produce visually different outputs."""
    img_a = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    # Image B with sharp edges — half black, half white
    img_b = np.zeros((100, 100, 3), dtype=np.uint8)
    img_b[:, 50:, :] = 255

    result_low = compute_hybrid(img_a, img_b, sigma_low=5.0, sigma_high=1.0)
    result_high = compute_hybrid(img_a, img_b, sigma_low=5.0, sigma_high=10.0)

    # The two results should differ meaningfully
    diff = np.abs(result_low.astype(float) - result_high.astype(float)).mean()
    assert diff > 1.0, (
        f"Expected significant difference between sigma_high=1 and 10, got mean diff={diff:.2f}"
    )


# ───────────────────────────────────────────
# TEST 3 — No Mutation of Originals
# ───────────────────────────────────────────
def test_no_mutation():
    """The original input arrays must not be modified by compute_hybrid."""
    img_a = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    img_b = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)

    # Keep copies to compare after computation
    a_copy = img_a.copy()
    b_copy = img_b.copy()

    _ = compute_hybrid(img_a, img_b, sigma_low=5.0, sigma_high=5.0)

    np.testing.assert_array_equal(img_a, a_copy, err_msg="Image A was mutated!")
    np.testing.assert_array_equal(img_b, b_copy, err_msg="Image B was mutated!")
