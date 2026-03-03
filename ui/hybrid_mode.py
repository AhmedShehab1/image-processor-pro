"""
HybridModeWidget — Thin UI layer for Perceptual Hybrid Images.

All frequency-domain logic (Gaussian masks, FFT/IFFT, DC removal, energy
balancing, adaptive sigma, blending) lives exclusively in
core.operations.HybridImage.

This widget is responsible ONLY for:
    - Loading images
    - Instantiating HybridImage
    - Calling public methods (apply / apply_extended)
    - Caching returned components
    - Viewing-distance simulation (reweighting cached float components)
    - Displaying results
    - Undo / redo
"""

import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QSlider, QSizePolicy
)
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import Qt, pyqtSignal

from ui.clickable_image_label import ClickableImageLabel
from core.operations import HybridImage


class HybridModeWidget(QWidget):
    """
    Top-level hybrid images workspace — pure grayscale pipeline.

    Delegates frequency-domain extraction to HybridImage (core.operations)
    and adds viewing distance simulation.

    Signals:
        hybrid_computed(np.ndarray): Emitted when a new hybrid_display is
            ready so the main window can update the histogram panel.
    """

    hybrid_computed = pyqtSignal(np.ndarray)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, parent=None):
        super().__init__(parent)

        # ---- State (persists across tab switches) ----
        self.originalA = None          # uint8 — as loaded (BGR or gray)
        self.originalB = None          # uint8 — as loaded (BGR or gray)
        self.hybrid_display = None     # uint8 — after viewing-distance simulation

        # Cached float64 components from HybridImage._pipeline()
        self._cached_low = None        # float64 — LPF(A) after energy balancing
        self._cached_high = None       # float64 — HPF(B) after energy balancing

        # ---- Undo / Redo ----
        self._undo_stack: list[dict] = []
        self._redo_stack: list[dict] = []
        self._MAX_HISTORY = 20

        self._setup_ui()

    # ------------------------------------------------------------------
    # UI Setup
    # ------------------------------------------------------------------

    def _setup_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        # ---- Top Row: Image A | Image B ----
        top_row = QHBoxLayout()
        top_row.setSpacing(16)

        # -- Left panel: Image A --
        left = QVBoxLayout()
        left.setSpacing(4)

        title_a = QLabel("Image A — Low Frequency Source")
        title_a.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_a.setStyleSheet("color: #00BCD4; font-weight: bold; font-size: 12px;")
        left.addWidget(title_a)

        self.label_a = ClickableImageLabel("Click to load Image A")
        self.label_a.image_loaded.connect(self._on_image_a_loaded)
        left.addWidget(self.label_a, stretch=1)

        top_row.addLayout(left, stretch=1)

        # -- Right panel: Image B --
        right = QVBoxLayout()
        right.setSpacing(4)

        title_b = QLabel("Image B — High Frequency Source")
        title_b.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_b.setStyleSheet("color: #FF9800; font-weight: bold; font-size: 12px;")
        right.addWidget(title_b)

        self.label_b = ClickableImageLabel("Click to load Image B")
        self.label_b.image_loaded.connect(self._on_image_b_loaded)
        right.addWidget(self.label_b, stretch=1)

        top_row.addLayout(right, stretch=1)
        root.addLayout(top_row, stretch=1)

        # ---- Sigma info label ----
        self.sigma_info = QLabel("Adaptive σ: load both images and process")
        self.sigma_info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.sigma_info.setStyleSheet(
            "color: #9E9E9E; font-size: 11px; font-style: italic;"
        )
        root.addWidget(self.sigma_info)

        # ---- Process + Reset buttons ----
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        self.process_btn = QPushButton("▶ Process Hybrid")
        self.process_btn.setObjectName("process_btn")
        self.process_btn.setMinimumHeight(38)
        self.process_btn.clicked.connect(self.process_hybrid)
        btn_row.addWidget(self.process_btn)

        self.reset_btn = QPushButton("↺ Reset")
        self.reset_btn.setMinimumHeight(38)
        self.reset_btn.clicked.connect(self._reset_perception)
        btn_row.addWidget(self.reset_btn)
        btn_row.addStretch()
        root.addLayout(btn_row)

        # ---- Hybrid result ----
        result_title = QLabel("Hybrid Result")
        result_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        result_title.setStyleSheet("color: #AB47BC; font-weight: bold; font-size: 13px;")
        root.addWidget(result_title)

        self.label_result = QLabel("Process to generate hybrid")
        self.label_result.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label_result.setObjectName("canvas_placeholder")
        self.label_result.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.label_result.setMinimumSize(1, 1)
        self.label_result.setStyleSheet("border: 1px solid #555; border-radius: 4px;")
        root.addWidget(self.label_result, stretch=1)

        # ---- Viewing distance slider ----
        dist_row = QHBoxLayout()
        dist_row.addWidget(QLabel("Viewing Distance:"))
        near_lbl = QLabel("Near")
        near_lbl.setStyleSheet("color: #4CAF50; font-size: 10px;")
        dist_row.addWidget(near_lbl)
        self.distance_slider = QSlider(Qt.Orientation.Horizontal)
        self.distance_slider.setRange(0, 100)
        self.distance_slider.setValue(0)
        self.distance_slider.valueChanged.connect(self._simulate_viewing_distance)
        dist_row.addWidget(self.distance_slider, stretch=1)
        far_lbl = QLabel("Far")
        far_lbl.setStyleSheet("color: #F44336; font-size: 10px;")
        dist_row.addWidget(far_lbl)
        self.distance_val = QLabel("0")
        self.distance_val.setStyleSheet("font-weight: bold; min-width: 25px;")
        dist_row.addWidget(self.distance_val)
        root.addLayout(dist_row)

        # ---- Undo / Redo ----
        undo_row = QHBoxLayout()
        self.undo_btn = QPushButton("↩ Undo")
        self.redo_btn = QPushButton("↪ Redo")
        self.undo_btn.setEnabled(False)
        self.redo_btn.setEnabled(False)
        self.undo_btn.clicked.connect(self._perform_undo)
        self.redo_btn.clicked.connect(self._perform_redo)
        undo_row.addWidget(self.undo_btn)
        undo_row.addWidget(self.redo_btn)
        undo_row.addStretch()
        root.addLayout(undo_row)

    # ------------------------------------------------------------------
    # Small UI helpers
    # ------------------------------------------------------------------

    def _reset_perception(self):
        """Reset viewing distance slider."""
        self.distance_slider.setValue(0)

    # ------------------------------------------------------------------
    # Image Loading
    # ------------------------------------------------------------------

    def _on_image_a_loaded(self, img: np.ndarray):
        """Store Image A and display it in its slot."""
        self.originalA = img.copy()
        self.label_a.set_image(img)

    def _on_image_b_loaded(self, img: np.ndarray):
        """Store Image B and display it in its slot."""
        self.originalB = img.copy()
        self.label_b.set_image(img)

    # ------------------------------------------------------------------
    # Hybrid Computation (Process button only)
    # ------------------------------------------------------------------

    def process_hybrid(self):
        """
        Compute the hybrid image by delegating entirely to HybridImage.
        All frequency-domain processing happens inside apply_extended().
        """
        if self.originalA is None or self.originalB is None:
            return

        self._push_undo()

        # Instantiate and run via public API only
        op = HybridImage(
            image_high=self.originalB,
            alpha=1.0,
            beta=1.0,
            grayscale_preview=True,
        )
        results = op.apply_extended(self.originalA)

        # Cache float64 intermediates for viewing-distance reweighting
        self._cached_low = op.cache_low
        self._cached_high = op.cache_high

        # Show effective adaptive sigmas returned by HybridImage
        sigmas = results["effective_sigmas"]
        self.sigma_info.setText(
            f"Adaptive σ:  LP = {sigmas['lp']:.1f}   |   "
            f"HP = {sigmas['hp']:.1f}"
        )

        # Apply current viewing distance
        self._apply_distance()

    # ------------------------------------------------------------------
    # Perceptual Viewing Distance Simulation
    # ------------------------------------------------------------------

    def _simulate_viewing_distance(self):
        """
        Slider changed → recompute hybrid_display.

        Viewing distance is simulated by attenuating high-frequency
        energy:  high_attenuation = (1 - distance_ratio)^2
        The low component is NEVER modified.
        """
        d = self.distance_slider.value()
        self.distance_val.setText(str(d))
        self._apply_distance()

    def _apply_distance(self):
        """
        Reconstruct hybrid_display by scaling ONLY the cached high-frequency
        component.  The low-frequency component is NEVER modified.

        At distance 0   → high_attenuation = 1.0 (full detail)
        At distance 100 → high_attenuation = 0.0 (pure low-freq image)
        """
        if self._cached_low is None or self._cached_high is None:
            return

        d = self.distance_slider.value()
        distance_ratio = d / 100.0

        # Quadratic attenuation — only scales the high component
        high_attenuation = (1.0 - distance_ratio) ** 2

        # Reweight cached float components; clip to valid uint8 range
        display = self._cached_low + high_attenuation * self._cached_high
        self.hybrid_display = np.clip(display, 0, 255).astype(np.uint8)

        self._display_result(self.hybrid_display)
        self.hybrid_computed.emit(self.hybrid_display)

    def _display_result(self, cv_img: np.ndarray):
        """Convert a grayscale image to QPixmap and display it."""
        if cv_img is None:
            return
        if len(cv_img.shape) == 2:
            h, w = cv_img.shape
            img = np.ascontiguousarray(cv_img)
            q_img = QImage(img.tobytes(), w, h, w, QImage.Format.Format_Grayscale8)
        else:
            h, w, ch = cv_img.shape
            rgb = np.ascontiguousarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))
            q_img = QImage(rgb.tobytes(), w, h, ch * w, QImage.Format.Format_RGB888)

        pixmap = QPixmap.fromImage(q_img)
        scaled = pixmap.scaled(
            self.label_result.size(), Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.label_result.setPixmap(scaled)

    # ------------------------------------------------------------------
    # Undo / Redo
    # ------------------------------------------------------------------

    def _snapshot(self) -> dict:
        """Capture the current hybrid state for undo/redo."""
        return {
            "distance": self.distance_slider.value(),
            "cached_low": self._cached_low.copy() if self._cached_low is not None else None,
            "cached_high": self._cached_high.copy() if self._cached_high is not None else None,
            "hybrid_display": self.hybrid_display.copy() if self.hybrid_display is not None else None,
        }

    def _restore(self, snap: dict):
        """Restore a previously captured state."""
        # Block signals to prevent _simulate_viewing_distance from firing
        # with stale cached data while we're mid-restore
        self.distance_slider.blockSignals(True)
        self.distance_slider.setValue(snap["distance"])
        self.distance_val.setText(str(snap["distance"]))
        self.distance_slider.blockSignals(False)

        self._cached_low = snap["cached_low"]
        self._cached_high = snap["cached_high"]
        self.hybrid_display = snap["hybrid_display"]

        if self._cached_low is not None:
            self._apply_distance()
        else:
            # Undone to pre-process state — clear the result display
            self.label_result.clear()
            self.label_result.setText("Process to generate hybrid")
            self.hybrid_display = None

        self._update_undo_buttons()

    def _push_undo(self):
        """Push current state to undo stack before a destructive action."""
        self._undo_stack.append(self._snapshot())
        if len(self._undo_stack) > self._MAX_HISTORY:
            self._undo_stack.pop(0)
        self._redo_stack.clear()
        self._update_undo_buttons()

    def _perform_undo(self):
        if not self._undo_stack:
            return
        self._redo_stack.append(self._snapshot())
        prev = self._undo_stack.pop()
        self._restore(prev)

    def _perform_redo(self):
        if not self._redo_stack:
            return
        self._undo_stack.append(self._snapshot())
        nxt = self._redo_stack.pop()
        self._restore(nxt)

    def _update_undo_buttons(self):
        self.undo_btn.setEnabled(len(self._undo_stack) > 0)
        self.redo_btn.setEnabled(len(self._redo_stack) > 0)

    # ------------------------------------------------------------------
    # Resize
    # ------------------------------------------------------------------

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.hybrid_display is not None:
            self._display_result(self.hybrid_display)
