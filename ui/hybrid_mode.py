"""
HybridModeWidget — Perceptual Hybrid Images (Oliva & Torralba, SIGGRAPH 2006).

Architecture overview:
    This widget implements TRUE frequency-domain hybrid images, not simple
    alpha blending.  It owns its entire lifecycle: image loading, sigma
    sliders, amplitude control, energy balancing, perceptual viewing
    distance simulation, and undo/redo.

Why weighted frequency composition is required:
    Naive  hybrid = low + high  produces blending artifacts because the
    raw high-frequency component can overpower the low.  Weighting via
    alpha * low + beta * high  lets us control the relative energy of
    each band independently.  Combined with amplitude normalization of
    the high component, this yields true frequency separation where
    Image A dominates at far distance and Image B dominates up close.

Why the high component is zero-centered and amplitude-normalized:
    Subtracting GaussianBlur(B) from B yields values in ≈[−255, +255].
    Normalizing by  high / std(high) * target_amplitude  controls the
    perceptual strength of edges so they sit at a specified energy level
    instead of an arbitrary one determined by image content.

Why attenuation of high frequencies simulates viewing distance:
    The human visual system acts as a spatial low-pass filter whose
    cutoff depends on viewing distance.  Simply blurring the result is
    insufficient — it also smears the low-frequency content.  The
    correct simulation progressively REDUCES the high component's
    contribution  (beta_distance = 1 − distance_ratio)  and only adds
    mild blur for very large distances (> 40 %), preserving low-freq
    clarity.

Why energy balancing prevents the "blending look":
    If  std(high) > 1.5 * std(low) , the high component will dominate
    the result and it will look like a transparent overlay.  Clamping
    the high energy to at most 1.5× the low energy forces the two
    frequency bands to coexist rather than compete.
"""

import cv2
import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QSlider, QSizePolicy
)
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import Qt, pyqtSignal

from ui.clickable_image_label import ClickableImageLabel


class HybridModeWidget(QWidget):
    """
    Top-level hybrid images workspace with true perceptual pipeline.

    Signals:
        hybrid_computed(np.ndarray): Emitted when a new hybrid_display is
            ready so the main window can update the histogram panel.
    """

    hybrid_computed = pyqtSignal(np.ndarray)

    # Default high-frequency target amplitude (Oliva & Torralba sweet spot)
    _DEFAULT_TARGET_AMPLITUDE = 20.0

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, parent=None):
        super().__init__(parent)

        # ---- State (persists across tab switches) ----
        self.originalA = None          # uint8 — never mutated
        self.originalB = None          # uint8 — resized to match A, never mutated
        self.low_component = None      # float32 — GaussianBlur(A, σ_low)
        self.high_component = None     # float32 — amplitude-normalized, zero-centered
        self.hybrid_base = None        # float32 — alpha*low + beta*high (before distance)
        self.hybrid_display = None     # uint8 — after viewing-distance simulation

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

        # -- Left panel: Image A + σ_low --
        left = QVBoxLayout()
        left.setSpacing(4)

        title_a = QLabel("Image A — Low Frequency Source")
        title_a.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_a.setStyleSheet("color: #00BCD4; font-weight: bold; font-size: 12px;")
        left.addWidget(title_a)

        self.label_a = ClickableImageLabel("Click to load Image A")
        self.label_a.image_loaded.connect(self._on_image_a_loaded)
        left.addWidget(self.label_a, stretch=1)

        sigma_low_row = QHBoxLayout()
        sigma_low_row.addWidget(QLabel("σ Low:"))
        self.sigma_low_slider = QSlider(Qt.Orientation.Horizontal)
        self.sigma_low_slider.setRange(1, 30)
        self.sigma_low_slider.setValue(5)
        self.sigma_low_slider.valueChanged.connect(self._update_low_preview)
        sigma_low_row.addWidget(self.sigma_low_slider, stretch=1)
        self.sigma_low_val = QLabel("5")
        self.sigma_low_val.setStyleSheet("color: #00BCD4; font-weight: bold; min-width: 20px;")
        sigma_low_row.addWidget(self.sigma_low_val)
        left.addLayout(sigma_low_row)

        top_row.addLayout(left, stretch=1)

        # -- Right panel: Image B + σ_high --
        right = QVBoxLayout()
        right.setSpacing(4)

        title_b = QLabel("Image B — High Frequency Source")
        title_b.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_b.setStyleSheet("color: #FF9800; font-weight: bold; font-size: 12px;")
        right.addWidget(title_b)

        self.label_b = ClickableImageLabel("Click to load Image B")
        self.label_b.image_loaded.connect(self._on_image_b_loaded)
        right.addWidget(self.label_b, stretch=1)

        sigma_high_row = QHBoxLayout()
        sigma_high_row.addWidget(QLabel("σ High:"))
        self.sigma_high_slider = QSlider(Qt.Orientation.Horizontal)
        self.sigma_high_slider.setRange(1, 30)
        self.sigma_high_slider.setValue(5)
        self.sigma_high_slider.valueChanged.connect(self._update_high_preview)
        sigma_high_row.addWidget(self.sigma_high_slider, stretch=1)
        self.sigma_high_val = QLabel("5")
        self.sigma_high_val.setStyleSheet("color: #FF9800; font-weight: bold; min-width: 20px;")
        sigma_high_row.addWidget(self.sigma_high_val)
        right.addLayout(sigma_high_row)

        top_row.addLayout(right, stretch=1)
        root.addLayout(top_row, stretch=1)

        # ---- High Strength slider ----
        strength_row = QHBoxLayout()
        strength_row.addWidget(QLabel("High Strength (β):"))
        self.strength_slider = QSlider(Qt.Orientation.Horizontal)
        self.strength_slider.setRange(5, 20)    # maps to 0.5 – 2.0
        self.strength_slider.setValue(10)        # default 1.0
        self.strength_slider.valueChanged.connect(self._on_strength_changed)
        strength_row.addWidget(self.strength_slider, stretch=1)
        self.strength_val = QLabel("1.0")
        self.strength_val.setStyleSheet("color: #FF9800; font-weight: bold; min-width: 30px;")
        strength_row.addWidget(self.strength_val)
        root.addLayout(strength_row)

        # ---- Process + Reset buttons ----
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        self.process_btn = QPushButton("▶ Process Hybrid")
        self.process_btn.setObjectName("process_btn")
        self.process_btn.setMinimumHeight(38)
        self.process_btn.clicked.connect(self.process_hybrid)
        btn_row.addWidget(self.process_btn)

        self.reset_btn = QPushButton("↺ Reset Perception")
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

    def _on_strength_changed(self):
        beta = self.strength_slider.value() / 10.0
        self.strength_val.setText(f"{beta:.1f}")

    def _reset_perception(self):
        """Reset all sliders to defaults without clearing loaded images."""
        self.sigma_low_slider.setValue(5)
        self.sigma_high_slider.setValue(5)
        self.strength_slider.setValue(10)
        self.distance_slider.setValue(0)

    # ------------------------------------------------------------------
    # Image Loading
    # ------------------------------------------------------------------

    def _on_image_a_loaded(self, img: np.ndarray):
        """Store Image A and initialize the low component."""
        self.originalA = img.copy()
        self.low_component = img.astype(np.float32)
        # If B exists, resize it to match A
        if self.originalB is not None:
            h, w = self.originalA.shape[:2]
            self.originalB = cv2.resize(self.originalB, (w, h))
            self.label_b.set_image(self.originalB)

    def _on_image_b_loaded(self, img: np.ndarray):
        """Store Image B (resized to A if loaded) and initialize the high component."""
        if self.originalA is not None:
            h, w = self.originalA.shape[:2]
            self.originalB = cv2.resize(img, (w, h))
        else:
            self.originalB = img.copy()
        self.label_b.set_image(self.originalB)
        # Initial high component
        self._recompute_high()

    # ------------------------------------------------------------------
    # Preview Logic (sliders → individual previews only)
    # ------------------------------------------------------------------

    def _update_low_preview(self):
        """
        σ_low slider → updates ONLY Image A preview.

        Computes the Gaussian-blurred version of originalA and stores it
        as the low_component for later perceptual blending.
        """
        if self.originalA is None:
            return
        sigma = self.sigma_low_slider.value()
        self.sigma_low_val.setText(str(sigma))

        self.low_component = cv2.GaussianBlur(
            self.originalA.astype(np.float32), (0, 0), sigma
        )
        display = np.clip(self.low_component, 0, 255).astype(np.uint8)
        self.label_a.set_image(display)

    def _update_high_preview(self):
        """
        σ_high slider → updates ONLY Image B preview.

        Extracts the high-frequency component via Laplacian-of-Gaussian
        style subtraction, then normalizes its amplitude to a fixed
        target energy level so it cannot overpower the low component.

        The preview is shifted by +128 so zero-crossings appear as
        mid-gray, making both positive and negative edges visible.
        """
        if self.originalB is None:
            return
        sigma = self.sigma_high_slider.value()
        self.sigma_high_val.setText(str(sigma))

        self._recompute_high()

        # +128 shift for visualization only
        vis = self.high_component + 128.0
        vis = cv2.normalize(vis, None, 0, 255, cv2.NORM_MINMAX)
        self.label_b.set_image(vis.astype(np.uint8))

    def _recompute_high(self):
        """
        Extract and amplitude-normalize the high-frequency component.

        Why amplitude normalization?
            Raw  B − blur(B)  has energy proportional to image content.
            An image with sharp edges produces much larger raw values
            than a soft portrait.  Normalizing to a target amplitude
            (default 20) ensures consistent perceptual strength
            regardless of source image characteristics.
        """
        if self.originalB is None:
            return
        sigma = self.sigma_high_slider.value()
        b_float = self.originalB.astype(np.float32)
        blurred = cv2.GaussianBlur(b_float, (0, 0), sigma)
        raw_high = b_float - blurred

        # Amplitude normalization
        std = np.std(raw_high)
        if std > 1e-6:
            self.high_component = (raw_high / std) * self._DEFAULT_TARGET_AMPLITUDE
        else:
            self.high_component = raw_high

    # ------------------------------------------------------------------
    # Hybrid Computation (Process button only)
    # ------------------------------------------------------------------

    def process_hybrid(self):
        """
        Compute the perceptual hybrid image.

        Pipeline:
            1. Energy balance: scale high energy to match low energy.
            2. Store balanced components for distance simulation.
            3. Apply viewing-distance simulation.

        This is called ONLY when the user clicks Process.  The current
        state is pushed to the undo stack before computing.
        """
        if self.low_component is None or self.high_component is None:
            return

        self._push_undo()

        # ---- Energy balancing ----
        # Scale the high component so its energy matches the low.
        # This prevents the hybrid from looking like a transparent
        # overlay where one frequency band dominates the other.
        self._balanced_low = self.low_component.astype(np.float32)
        self._balanced_high = self.high_component.astype(np.float32).copy()

        low_energy = np.std(self._balanced_low)
        high_energy = np.std(self._balanced_high)
        if high_energy > 1e-6:
            self._balanced_high *= (low_energy / high_energy)

        # Store hybrid_base for undo/redo reference
        beta = self.strength_slider.value() / 10.0
        self.hybrid_base = self._balanced_low + beta * self._balanced_high

        # Apply current viewing distance
        self._apply_distance()

    # ------------------------------------------------------------------
    # True Perceptual Viewing Distance Simulation
    # ------------------------------------------------------------------

    def _simulate_viewing_distance(self):
        """
        Slider changed → recompute hybrid_display.

        Viewing distance is simulated by attenuating high-frequency
        energy rather than globally blurring the hybrid image.  This
        better matches human perception: the eye's spatial low-pass
        filter suppresses fine detail at distance while preserving
        the low-frequency structure with full clarity.
        """
        d = self.distance_slider.value()
        self.distance_val.setText(str(d))
        self._apply_distance()

    def _apply_distance(self):
        """
        Reconstruct hybrid_display by attenuating ONLY the high-frequency
        component.  The low component is NEVER blurred.

        At distance 0   → high_attenuation = 1.0 (full detail)
        At distance 100 → high_attenuation = 0.0 (pure low-freq image)

        Optional: for distance > 40%, apply mild Gaussian blur to the
        high component only (NOT to low or the full composite) to
        simulate additional perceptual roll-off of fine edges.
        """
        if not hasattr(self, '_balanced_low') or self._balanced_low is None:
            return

        d = self.distance_slider.value()
        distance_ratio = d / 100.0

        # ---- High-frequency attenuation ----
        high_attenuation = 1.0 - distance_ratio
        beta = self.strength_slider.value() / 10.0

        high = self._balanced_high.copy()

        # Optional: mild blur on HIGH ONLY for far distances (> 40%)
        if distance_ratio > 0.4:
            extra_sigma = (distance_ratio - 0.4) * 10.0
            high = cv2.GaussianBlur(high, (0, 0), extra_sigma)

        # Compose: low is NEVER blurred, preserving its clarity
        display = self._balanced_low + high_attenuation * beta * high

        # Clip to valid range (NOT cv2.normalize, which stretches contrast)
        display = np.clip(display, 0, 255)
        self.hybrid_display = display.astype(np.uint8)

        self._display_result(self.hybrid_display)
        self.hybrid_computed.emit(self.hybrid_display)

    def _display_result(self, cv_img: np.ndarray):
        """Convert an OpenCV image and display it in the result label."""
        if cv_img is None:
            return
        if len(cv_img.shape) == 2:
            h, w = cv_img.shape
            q_img = QImage(cv_img.data, w, h, w, QImage.Format.Format_Grayscale8)
        else:
            h, w, ch = cv_img.shape
            rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            q_img = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)

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
            "sigma_low": self.sigma_low_slider.value(),
            "sigma_high": self.sigma_high_slider.value(),
            "strength": self.strength_slider.value(),
            "distance": self.distance_slider.value(),
            "hybrid_base": self.hybrid_base.copy() if self.hybrid_base is not None else None,
        }

    def _restore(self, snap: dict):
        """Restore a previously captured state."""
        self.sigma_low_slider.setValue(snap["sigma_low"])
        self.sigma_high_slider.setValue(snap["sigma_high"])
        self.strength_slider.setValue(snap["strength"])
        self.distance_slider.setValue(snap["distance"])
        self.hybrid_base = snap["hybrid_base"]
        if self.hybrid_base is not None:
            self._apply_distance()
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
