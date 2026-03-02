"""
Tests for the corrected hybrid perception pipeline.

Validates:
    1. High-freq attenuation: near shows full detail, far shows only low.
    2. Variance: near > far (high frequencies add variance).
    3. Energy balance: high energy matches low energy after balancing.
    4. No mutation: originals unchanged after full pipeline.
"""

import cv2
import numpy as np
import pytest

TARGET_AMPLITUDE = 20.0


# ───────────────────────────────────────────
# Helpers — replicate HybridModeWidget math
# ───────────────────────────────────────────

def compute_low(img_a: np.ndarray, sigma: float) -> np.ndarray:
    return cv2.GaussianBlur(img_a.astype(np.float32), (0, 0), sigma)


def compute_high_normalized(img_b: np.ndarray, sigma: float) -> np.ndarray:
    b = img_b.astype(np.float32)
    blurred = cv2.GaussianBlur(b, (0, 0), sigma)
    raw = b - blurred
    std = np.std(raw)
    if std > 1e-6:
        return (raw / std) * TARGET_AMPLITUDE
    return raw


def energy_balance(low: np.ndarray, high: np.ndarray) -> np.ndarray:
    """Unconditional energy balance: scale high to match low energy."""
    h = high.copy()
    high_e = np.std(h)
    low_e = np.std(low)
    if high_e > 1e-6:
        h *= (low_e / high_e)
    return h


def compose_at_distance(low, balanced_high, beta, distance):
    """
    Correct perceptual model: attenuate ONLY high, optional blur on high ONLY.
    Low component is NEVER blurred.
    """
    ratio = distance / 100.0
    high_attenuation = 1.0 - ratio

    high = balanced_high.copy()

    # Optional mild blur on HIGH ONLY for far distances
    if ratio > 0.4:
        extra_sigma = (ratio - 0.4) * 10.0
        high = cv2.GaussianBlur(high, (0, 0), extra_sigma)

    display = low + high_attenuation * beta * high
    display = np.clip(display, 0, 255)
    return display.astype(np.uint8)


# ───────────────────────────────────────────
# TEST 1 — High-Freq Attenuation With Distance
# ───────────────────────────────────────────

def test_high_freq_attenuated_with_distance():
    """
    At distance=100 the result must be pure low (no high contribution).
    At distance=0 the result must include high-frequency detail.
    """
    np.random.seed(42)
    img_a = np.random.randint(50, 200, (100, 100, 3), dtype=np.uint8)
    img_b = np.zeros((100, 100, 3), dtype=np.uint8)
    img_b[:, 50:, :] = 255

    low = compute_low(img_a, 5.0)
    high = compute_high_normalized(img_b, 5.0)
    balanced = energy_balance(low, high)

    near = compose_at_distance(low, balanced, beta=1.0, distance=0)
    far = compose_at_distance(low, balanced, beta=1.0, distance=100)

    # Near composite differs from low (has high contribution)
    diff_near = np.abs(near.astype(float) - low).mean()
    # Far composite should be identical to clipped low (attenuation = 0)
    low_clipped = np.clip(low, 0, 255).astype(np.uint8)
    diff_far = np.abs(far.astype(float) - low_clipped.astype(float)).mean()

    assert diff_near > diff_far, (
        f"Near diff ({diff_near:.2f}) should exceed far diff ({diff_far:.2f})"
    )
    assert diff_far < 1.0, f"At distance=100, result should equal low (diff={diff_far:.2f})"


# ───────────────────────────────────────────
# TEST 2 — Variance Decreases With Distance
# ───────────────────────────────────────────

def test_variance_decreases_with_distance():
    """Near must have higher variance than far (high-freq adds detail)."""
    np.random.seed(42)
    img_a = np.random.randint(50, 200, (100, 100, 3), dtype=np.uint8)
    img_b = np.zeros((100, 100, 3), dtype=np.uint8)
    img_b[:, 50:, :] = 255

    low = compute_low(img_a, 5.0)
    high = compute_high_normalized(img_b, 5.0)
    balanced = energy_balance(low, high)

    var_0 = compose_at_distance(low, balanced, 1.0, 0).astype(float).var()
    var_100 = compose_at_distance(low, balanced, 1.0, 100).astype(float).var()

    assert var_0 > var_100, (
        f"Near variance ({var_0:.1f}) should exceed far ({var_100:.1f})"
    )


# ───────────────────────────────────────────
# TEST 3 — Energy Balance
# ───────────────────────────────────────────

def test_energy_balance():
    """After balancing, high std should match low std."""
    np.random.seed(42)
    img_a = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    img_b = np.zeros((100, 100, 3), dtype=np.uint8)
    img_b[:, 50:, :] = 255

    low = compute_low(img_a, 5.0)
    high = compute_high_normalized(img_b, 5.0)
    balanced = energy_balance(low, high)

    low_e = np.std(low)
    balanced_e = np.std(balanced)

    assert abs(balanced_e - low_e) < 1.0, (
        f"Balanced energy ({balanced_e:.1f}) should match low ({low_e:.1f})"
    )


# ───────────────────────────────────────────
# TEST 4 — No Mutation
# ───────────────────────────────────────────

def test_no_mutation():
    """Original images must remain unchanged after full pipeline."""
    img_a = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    img_b = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    a_copy = img_a.copy()
    b_copy = img_b.copy()

    low = compute_low(img_a, 5.0)
    high = compute_high_normalized(img_b, 5.0)
    balanced = energy_balance(low, high)
    _ = compose_at_distance(low, balanced, 1.0, 50)

    np.testing.assert_array_equal(img_a, a_copy, err_msg="Image A was mutated!")
    np.testing.assert_array_equal(img_b, b_copy, err_msg="Image B was mutated!")
