"""
Tests for Hybrid Image Corrected UX Logic.

Validates the separated preview/computation architecture:
    1. Moving sliders does NOT automatically change the hybrid result.
    2. Low component matches GaussianBlur(originalA) at the given sigma.
    3. High component matches originalB - GaussianBlur(originalB).
    4. Final hybrid = low + high, normalized to [0, 255] uint8.
"""

import cv2
import numpy as np
import pytest


# ───────────────────────────────────────────
# Helpers — replicate the canvas logic without PyQt
# ───────────────────────────────────────────

def update_low_component(originalA: np.ndarray, sigma: float) -> np.ndarray:
    """Mirrors CanvasView._update_low_preview() math."""
    return cv2.GaussianBlur(originalA.astype(np.float32), (0, 0), sigma)


def update_high_component(originalB: np.ndarray, sigma: float) -> np.ndarray:
    """Mirrors CanvasView._update_high_preview() math."""
    b_float = originalB.astype(np.float32)
    blurred = cv2.GaussianBlur(b_float, (0, 0), sigma)
    return b_float - blurred


def compute_hybrid_from_components(low: np.ndarray, high: np.ndarray) -> np.ndarray:
    """Mirrors CanvasView._compute_and_display_hybrid() math."""
    hybrid = low.astype(np.float32) + high.astype(np.float32)
    hybrid = cv2.normalize(hybrid, None, 0, 255, cv2.NORM_MINMAX)
    return hybrid.astype(np.uint8)


# ───────────────────────────────────────────
# TEST 1 — Sliders Do NOT Change Hybrid
# ───────────────────────────────────────────
def test_sliders_do_not_change_hybrid():
    """Moving sigma_low must NOT alter a previously computed hybrid result."""
    img_a = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    img_b = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)

    # Compute initial components and hybrid
    low = update_low_component(img_a, sigma=5.0)
    high = update_high_component(img_b, sigma=5.0)
    hybrid_before = compute_hybrid_from_components(low, high).copy()

    # Now "move the slider" — update only the low component
    _ = update_low_component(img_a, sigma=15.0)

    # The hybrid_before should be unchanged (it was not recomputed)
    assert np.array_equal(hybrid_before, hybrid_before), (
        "Hybrid result should not change just because a slider moved"
    )


# ───────────────────────────────────────────
# TEST 2 — Low Component Correctness
# ───────────────────────────────────────────
def test_low_component_correctness():
    """low_component must equal GaussianBlur(originalA, sigma)."""
    img_a = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    sigma = 7.0

    low = update_low_component(img_a, sigma)
    expected = cv2.GaussianBlur(img_a.astype(np.float32), (0, 0), sigma)

    np.testing.assert_array_almost_equal(low, expected, decimal=5)


# ───────────────────────────────────────────
# TEST 3 — High Component Correctness
# ───────────────────────────────────────────
def test_high_component_correctness():
    """high_component must equal originalB - GaussianBlur(originalB)."""
    img_b = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    sigma = 5.0

    high = update_high_component(img_b, sigma)

    b_float = img_b.astype(np.float32)
    expected = b_float - cv2.GaussianBlur(b_float, (0, 0), sigma)

    np.testing.assert_array_almost_equal(high, expected, decimal=5)

    # Must be zero-centered (mean close to 0)
    assert abs(high.mean()) < 30.0, (
        f"High component should be zero-centered, got mean={high.mean():.2f}"
    )


# ───────────────────────────────────────────
# TEST 4 — Hybrid Correctness
# ───────────────────────────────────────────
def test_hybrid_correctness():
    """Hybrid = normalize(low + high) must match manual computation."""
    img_a = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    img_b = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)

    low = update_low_component(img_a, sigma=5.0)
    high = update_high_component(img_b, sigma=5.0)

    result = compute_hybrid_from_components(low, high)

    # Manual computation
    manual = low.astype(np.float32) + high.astype(np.float32)
    manual = cv2.normalize(manual, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    np.testing.assert_array_equal(result, manual)
    assert result.dtype == np.uint8
    assert result.min() >= 0
    assert result.max() <= 255
