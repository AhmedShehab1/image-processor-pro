"""
Tests for the grayscale hybrid pipeline using HybridImage from core.operations.

Validates:
    1. High-freq attenuation: near shows full detail, far shows only low.
    2. Variance: near > far (high frequencies add variance).
    3. Energy balance: balanced high std matches low std.
    4. No mutation: originals unchanged after full pipeline.
    5. Grayscale enforcement: all intermediates are 2D.
    6. DC bias removal: high component is zero-mean.
    7. Output range: clipped to [0, 255].
    8. HybridImage integration: apply_extended returns correct keys.
    9. Adaptive sigma: effective sigmas are computed from image diagonal.
   10. Center crop and resize: output is square with target size.
"""

import cv2
import numpy as np
import pytest
from core.operations import HybridImage, center_crop_and_resize


# ───────────────────────────────────────────
# Fixtures
# ───────────────────────────────────────────

@pytest.fixture
def grayscale_pair():
    """Return two 2D grayscale uint8 images."""
    np.random.seed(42)
    img_a = np.random.randint(50, 200, (100, 100), dtype=np.uint8)
    img_b = np.zeros((100, 100), dtype=np.uint8)
    img_b[:, 50:] = 255
    return img_a, img_b


@pytest.fixture
def hybrid_op(grayscale_pair):
    """Create a HybridImage operation with default settings."""
    img_a, img_b = grayscale_pair
    op = HybridImage(image_high=img_b, target_size=64)
    op._pipeline(img_a)
    return op


# ───────────────────────────────────────────
# Helpers — replicate viewing distance logic
# ───────────────────────────────────────────

def compose_at_distance(low, high, distance):
    """Attenuate high component with quadratic falloff."""
    ratio = distance / 100.0
    atten = (1.0 - ratio) ** 2
    display = low + atten * high
    shifted = display - display.min()
    return np.clip(shifted, 0, 255).astype(np.uint8)


# ───────────────────────────────────────────
# TEST 1 — High-Freq Attenuation With Distance
# ───────────────────────────────────────────

def test_high_freq_attenuated_with_distance(hybrid_op):
    """
    At distance=100 the result must be pure low (no high contribution).
    At distance=0 the result must include high-frequency detail.
    """
    low, high = hybrid_op.cache_low, hybrid_op.cache_high

    near = compose_at_distance(low, high, distance=0)
    far  = compose_at_distance(low, high, distance=100)

    # Near composite differs from "low only" more than far does
    low_only = compose_at_distance(low, high * 0, distance=0)
    diff_near = np.abs(near.astype(float) - low_only.astype(float)).mean()
    diff_far  = np.abs(far.astype(float)  - low_only.astype(float)).mean()

    assert diff_near > diff_far, (
        f"Near diff ({diff_near:.2f}) should exceed far diff ({diff_far:.2f})"
    )
    assert diff_far < 1.0, f"At distance=100, result should equal low (diff={diff_far:.2f})"


# ───────────────────────────────────────────
# TEST 2 — Variance Decreases With Distance
# ───────────────────────────────────────────

def test_variance_decreases_with_distance(hybrid_op):
    """Near must have higher variance than far (high-freq adds detail)."""
    low, high = hybrid_op.cache_low, hybrid_op.cache_high

    var_0   = compose_at_distance(low, high, 0).astype(float).var()
    var_100 = compose_at_distance(low, high, 100).astype(float).var()

    assert var_0 > var_100, (
        f"Near variance ({var_0:.1f}) should exceed far ({var_100:.1f})"
    )


# ───────────────────────────────────────────
# TEST 3 — Energy Balance
# ───────────────────────────────────────────

def test_energy_balance(hybrid_op):
    """After energy balancing, std(high) should approximately match std(low)."""
    low_e = np.std(hybrid_op.cache_low)
    high_e = np.std(hybrid_op.cache_high)

    # Energy balancing scales high to match low (within 20% tolerance)
    assert abs(high_e - low_e) / (low_e + 1e-8) < 0.2, (
        f"Balanced high energy ({high_e:.2f}) should be close to low ({low_e:.2f})"
    )


# ───────────────────────────────────────────
# TEST 4 — No Mutation
# ───────────────────────────────────────────

def test_no_mutation():
    """Original images must remain unchanged after full pipeline."""
    np.random.seed(42)
    img_a = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
    img_b = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
    a_copy = img_a.copy()
    b_copy = img_b.copy()

    op = HybridImage(image_high=img_b, target_size=64)
    _ = op.apply(img_a)

    np.testing.assert_array_equal(img_a, a_copy, err_msg="Image A was mutated!")
    np.testing.assert_array_equal(img_b, b_copy, err_msg="Image B was mutated!")


# ───────────────────────────────────────────
# TEST 5 — Grayscale Enforcement
# ───────────────────────────────────────────

def test_grayscale_intermediates(hybrid_op):
    """All cached intermediates must be 2D float64."""
    assert hybrid_op.cache_low.ndim == 2
    assert hybrid_op.cache_high.ndim == 2
    assert hybrid_op.cache_hybrid.ndim == 2
    assert hybrid_op.cache_low.dtype == np.float64
    assert hybrid_op.cache_high.dtype == np.float64


# ───────────────────────────────────────────
# TEST 6 — DC Bias Removal
# ───────────────────────────────────────────

def test_dc_bias_removed(hybrid_op):
    """High component must be zero-mean after DC bias removal."""
    assert abs(np.mean(hybrid_op.cache_high)) < 0.5, (
        f"High component mean should be near 0, got {np.mean(hybrid_op.cache_high):.4f}"
    )


# ───────────────────────────────────────────
# TEST 7 — Output Range
# ───────────────────────────────────────────

def test_output_range(grayscale_pair):
    """apply() result must be uint8 in [0, 255]."""
    img_a, img_b = grayscale_pair
    op = HybridImage(image_high=img_b, target_size=64)
    result = op.apply(img_a)

    assert result.dtype == np.uint8
    assert result.min() >= 0
    assert result.max() <= 255
    assert result.ndim == 2  # grayscale


# ───────────────────────────────────────────
# TEST 8 — apply_extended keys
# ───────────────────────────────────────────

def test_apply_extended_keys(grayscale_pair):
    """apply_extended must return all expected keys with correct types."""
    img_a, img_b = grayscale_pair
    op = HybridImage(image_high=img_b, target_size=64)
    result = op.apply_extended(img_a)

    for key in ("low_component", "high_component", "hybrid", "magnitude"):
        assert key in result, f"Missing key: {key}"
        assert result[key].ndim == 2, f"{key} should be 2D"
        assert result[key].dtype == np.uint8, f"{key} should be uint8"

    assert "effective_sigmas" in result
    assert "lp" in result["effective_sigmas"]
    assert "hp" in result["effective_sigmas"]

    # hybrid and magnitude should be identical
    np.testing.assert_array_equal(result["hybrid"], result["magnitude"])


# ───────────────────────────────────────────
# TEST 9 — Adaptive sigma
# ───────────────────────────────────────────

def test_adaptive_sigma(hybrid_op):
    """Effective sigmas should be positive and derived from image diagonal."""
    assert hybrid_op.effective_lp_sigma is not None
    assert hybrid_op.effective_hp_sigma is not None
    assert hybrid_op.effective_lp_sigma > 0
    assert hybrid_op.effective_hp_sigma > 0
    assert hybrid_op.effective_lp_sigma > hybrid_op.effective_hp_sigma


# ───────────────────────────────────────────
# TEST 10 — Center crop and resize
# ───────────────────────────────────────────

def test_center_crop_and_resize():
    """center_crop_and_resize should produce a square of target size."""
    img = np.random.randint(0, 256, (200, 300), dtype=np.uint8)
    result = center_crop_and_resize(img, size=64)
    assert result.shape == (64, 64)

    # Also works with 3-channel
    img_color = np.random.randint(0, 256, (200, 300, 3), dtype=np.uint8)
    result_color = center_crop_and_resize(img_color, size=64)
    assert result_color.shape == (64, 64, 3)
