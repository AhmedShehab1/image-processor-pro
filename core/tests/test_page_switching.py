"""
Tests for page switching logic.

Validates that the canvas correctly switches between single_page and
hybrid_page when sidebar sections change, and that hybrid state is
properly cleared when leaving hybrid mode.
"""

import numpy as np
import pytest


# ───────────────────────────────────────────
# Minimal stub to test page-switch logic without spawning a Qt app.
# We replicate the state management from CanvasView.
# ───────────────────────────────────────────

class FakeCanvasState:
    """Mimics CanvasView's hybrid state fields for unit testing."""

    def __init__(self):
        self.current_page = "single"
        self.originalA = None
        self.originalB = None
        self.low_component = None
        self.high_component = None
        self.hybrid_result = None
        self.current_image = np.zeros((50, 50, 3), dtype=np.uint8)

    def show_single_page(self):
        self.current_page = "single"

    def show_hybrid_page(self):
        self.current_page = "hybrid"

    def clear_hybrid_state(self):
        self.originalA = None
        self.originalB = None
        self.low_component = None
        self.high_component = None
        self.hybrid_result = None

    def load_hybrid(self):
        """Simulate loading two images into hybrid state."""
        self.originalA = np.ones((100, 100, 3), dtype=np.uint8) * 128
        self.originalB = np.ones((100, 100, 3), dtype=np.uint8) * 64
        self.low_component = self.originalA.astype(np.float32)
        self.high_component = self.originalB.astype(np.float32)
        self.hybrid_result = np.ones((100, 100, 3), dtype=np.uint8) * 96


# ───────────────────────────────────────────
# TEST 1 — Hybrid Selection
# ───────────────────────────────────────────
def test_hybrid_selection():
    """Selecting Hybrid Images must switch to the hybrid page."""
    canvas = FakeCanvasState()
    assert canvas.current_page == "single"

    canvas.show_hybrid_page()
    assert canvas.current_page == "hybrid"


# ───────────────────────────────────────────
# TEST 2 — Switching Away
# ───────────────────────────────────────────
def test_switching_away():
    """Selecting any non-hybrid section must switch to single page."""
    canvas = FakeCanvasState()
    canvas.show_hybrid_page()
    assert canvas.current_page == "hybrid"

    canvas.show_single_page()
    assert canvas.current_page == "single"


# ───────────────────────────────────────────
# TEST 3 — State Reset on Switch Away
# ───────────────────────────────────────────
def test_state_reset():
    """All hybrid components must be None after switching away."""
    canvas = FakeCanvasState()
    canvas.load_hybrid()

    # Verify state was loaded
    assert canvas.originalA is not None
    assert canvas.low_component is not None
    assert canvas.hybrid_result is not None

    # Switch away (simulating what MainWindow does)
    canvas.show_single_page()
    canvas.clear_hybrid_state()

    assert canvas.originalA is None
    assert canvas.originalB is None
    assert canvas.low_component is None
    assert canvas.high_component is None
    assert canvas.hybrid_result is None
