import cv2
import numpy as np
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                             QPushButton, QFileDialog, QStackedWidget, QGridLayout,
                             QSizePolicy, QSlider)
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import Qt, pyqtSignal

from core.operations import compute_hybrid


class CanvasView(QWidget):
    # Emit a signal when a new image is loaded so the MainWindow knows to update the Histogram
    image_loaded = pyqtSignal(np.ndarray)
    undo_requested = pyqtSignal()  # For future undo functionality
    redo_requested = pyqtSignal()  # For future redo functionality
    hybrid_result_ready = pyqtSignal(np.ndarray)  # Emitted when hybrid preview updates

    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_image = None
        self.original_image = None

        # Hybrid page state — strict separation of original / component / result
        self.originalA = None          # Original Image A (never mutated)
        self.originalB = None          # Original Image B (never mutated, pre-resized)
        self.low_component = None      # float32 — GaussianBlur(originalA)
        self.high_component = None     # float32 — originalB - GaussianBlur(originalB), zero-centered
        self.hybrid_result = None      # uint8 — final normalized hybrid

        self._setup_ui()

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)

        # --- Top Toolbar ---
        toolbar = QHBoxLayout()
        self.load_btn = QPushButton("📁 Load Image")
        self.load_btn.clicked.connect(self.load_image_dialog)

        self.reset_btn = QPushButton("⏪ Reset to Original")
        self.reset_btn.clicked.connect(self.reset_image)

        # --- ADD UNDO / REDO BUTTONS ---
        self.undo_btn = QPushButton("↩ Undo")
        self.redo_btn = QPushButton("↪ Redo")
        self.undo_btn.setEnabled(False)  # Initially disabled until we have history
        self.redo_btn.setEnabled(False)  # Initially disabled until we have history

        self.undo_btn.clicked.connect(self.undo_requested.emit)
        self.redo_btn.clicked.connect(self.redo_requested.emit)

        toolbar.addWidget(self.load_btn)
        toolbar.addWidget(self.reset_btn)
        toolbar.addWidget(self.undo_btn)
        toolbar.addWidget(self.redo_btn)
        toolbar.addStretch()
        main_layout.addLayout(toolbar)

        # --- Stacked Widget for Layout Routing ---
        # A StackedWidget allows us to place multiple layouts on top of each other
        # and flip between them like pages in a book.
        self.stack = QStackedWidget()
        main_layout.addWidget(self.stack, stretch=1)

        # 1. Single Image Page
        self.single_page = QWidget()
        single_layout = QVBoxLayout(self.single_page)
        self.single_label = QLabel("No Image Loaded")
        self.single_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.single_label.setObjectName("canvas_placeholder")
        single_layout.addWidget(self.single_label)
        self.single_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.single_label.setMinimumSize(1, 1)
        self.stack.addWidget(self.single_page)

        # 2. 4-Grid Edge Page (For Sobel, Prewitt, Roberts)
        self.grid_page = QWidget()
        grid_layout = QGridLayout(self.grid_page)
        grid_layout.setContentsMargins(10, 10, 10, 10)
        self.grid_orig = self._create_grid_label("Original")
        self.grid_x = self._create_grid_label("X Gradient")
        self.grid_y = self._create_grid_label("Y Gradient")
        self.grid_mag = self._create_grid_label("Magnitude")

        grid_layout.addWidget(self.grid_orig, 0, 0)
        grid_layout.addWidget(self.grid_x, 0, 1)
        grid_layout.addWidget(self.grid_y, 1, 0)
        grid_layout.addWidget(self.grid_mag, 1, 1)
        self.stack.addWidget(self.grid_page)

        # 3. Hybrid Images Page
        self._build_hybrid_page()

    def _build_hybrid_page(self):
        """
        Constructs the hybrid images canvas page with separated preview logic:

        ┌──────────────────────────────────────────────┐
        │  Image A (Low Freq)   │  Image B (High Freq)  │
        │  ─────────────────    │  ─────────────────     │
        │  σ Low: ■■■■■■■□□  5  │  σ High: ■■■■■□□□  5  │
        ├──────────────────────────────────────────────┤
        │  [ ▶ Process Hybrid ]                        │
        │           Hybrid Result                      │
        └──────────────────────────────────────────────┘

        UX contract:
            - σ_low slider  → updates ONLY Image A preview (low-pass blur).
            - σ_high slider → updates ONLY Image B preview (high-pass detail).
            - Hybrid result → updates ONLY when the user clicks Process.
        """
        self.hybrid_page = QWidget()
        page_layout = QVBoxLayout(self.hybrid_page)
        page_layout.setContentsMargins(8, 8, 8, 8)
        page_layout.setSpacing(8)

        # ---- Top Row: Image A | Image B ----
        top_row = QHBoxLayout()
        top_row.setSpacing(12)

        # -- Left Panel: Image A + σ_low slider --
        left_panel = QVBoxLayout()
        left_panel.setSpacing(4)

        title_a = QLabel("Image A — Low Frequency Source")
        title_a.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_a.setStyleSheet("color: #00BCD4; font-weight: bold; font-size: 11px;")
        left_panel.addWidget(title_a)

        self.hybrid_label_a = QLabel()
        self.hybrid_label_a.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.hybrid_label_a.setObjectName("canvas_placeholder")
        self.hybrid_label_a.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.hybrid_label_a.setMinimumSize(1, 1)
        self.hybrid_label_a.setStyleSheet("border: 1px solid #444; border-radius: 4px;")
        left_panel.addWidget(self.hybrid_label_a, stretch=1)

        sigma_low_row = QHBoxLayout()
        sigma_low_row.addWidget(QLabel("σ Low:"))
        self.hybrid_sigma_low = QSlider(Qt.Orientation.Horizontal)
        self.hybrid_sigma_low.setRange(1, 20)
        self.hybrid_sigma_low.setValue(5)
        self.hybrid_sigma_low.valueChanged.connect(self._update_low_preview)
        sigma_low_row.addWidget(self.hybrid_sigma_low, stretch=1)
        self.hybrid_sigma_low_val = QLabel("5")
        self.hybrid_sigma_low_val.setStyleSheet("color: #00BCD4; font-weight: bold; min-width: 20px;")
        sigma_low_row.addWidget(self.hybrid_sigma_low_val)
        left_panel.addLayout(sigma_low_row)

        top_row.addLayout(left_panel, stretch=1)

        # -- Right Panel: Image B + σ_high slider --
        right_panel = QVBoxLayout()
        right_panel.setSpacing(4)

        title_b = QLabel("Image B — High Frequency Source")
        title_b.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_b.setStyleSheet("color: #FF9800; font-weight: bold; font-size: 11px;")
        right_panel.addWidget(title_b)

        self.hybrid_label_b = QLabel()
        self.hybrid_label_b.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.hybrid_label_b.setObjectName("canvas_placeholder")
        self.hybrid_label_b.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.hybrid_label_b.setMinimumSize(1, 1)
        self.hybrid_label_b.setStyleSheet("border: 1px solid #444; border-radius: 4px;")
        right_panel.addWidget(self.hybrid_label_b, stretch=1)

        sigma_high_row = QHBoxLayout()
        sigma_high_row.addWidget(QLabel("σ High:"))
        self.hybrid_sigma_high = QSlider(Qt.Orientation.Horizontal)
        self.hybrid_sigma_high.setRange(1, 20)
        self.hybrid_sigma_high.setValue(5)
        self.hybrid_sigma_high.valueChanged.connect(self._update_high_preview)
        sigma_high_row.addWidget(self.hybrid_sigma_high, stretch=1)
        self.hybrid_sigma_high_val = QLabel("5")
        self.hybrid_sigma_high_val.setStyleSheet("color: #FF9800; font-weight: bold; min-width: 20px;")
        sigma_high_row.addWidget(self.hybrid_sigma_high_val)
        right_panel.addLayout(sigma_high_row)

        top_row.addLayout(right_panel, stretch=1)

        page_layout.addLayout(top_row, stretch=1)

        # ---- Bottom Row: Hybrid Result ----
        result_title = QLabel("Hybrid Result")
        result_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        result_title.setStyleSheet("color: #AB47BC; font-weight: bold; font-size: 12px;")
        page_layout.addWidget(result_title)

        self.hybrid_label_result = QLabel()
        self.hybrid_label_result.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.hybrid_label_result.setObjectName("canvas_placeholder")
        self.hybrid_label_result.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.hybrid_label_result.setMinimumSize(1, 1)
        self.hybrid_label_result.setStyleSheet("border: 1px solid #555; border-radius: 4px;")
        page_layout.addWidget(self.hybrid_label_result, stretch=1)

        self.stack.addWidget(self.hybrid_page)

    # ==========================================================
    # Page Switching  (prevents UI state leakage between modes)
    # ==========================================================

    def show_single_page(self):
        """
        Switch to the standard single-image canvas.

        Why explicit switching matters:
            Without explicit page selection, the QStackedWidget retains
            whichever page was last displayed.  If the user switches away
            from Hybrid Images in the sidebar, the hybrid layout would
            persist even though the controls no longer correspond to it.
            Explicit switching ensures every sidebar section owns its view.
        """
        self.stack.setCurrentWidget(self.single_page)
        if self.current_image is not None:
            self._set_scaled_pixmap(self.single_label, self.current_image)

    def show_hybrid_page(self):
        """
        Switch to the hybrid images canvas.

        The hybrid page must be isolated from the standard pipeline
        because it manages TWO source images and its own slider-driven
        preview cycle, which would conflict with the single-image
        display/undo/redo flow of every other section.
        """
        self.stack.setCurrentWidget(self.hybrid_page)

    def clear_hybrid_state(self):
        """
        Reset all hybrid-specific data so stale state cannot leak
        into a later hybrid session or interfere with single-image ops.
        """
        self.originalA = None
        self.originalB = None
        self.low_component = None
        self.high_component = None
        self.hybrid_result = None

        # Reset sliders to default
        self.hybrid_sigma_low.setValue(5)
        self.hybrid_sigma_high.setValue(5)

        # Clear preview labels
        self.hybrid_label_a.clear()
        self.hybrid_label_b.clear()
        self.hybrid_label_result.clear()

    # ==========================================================
    # Hybrid Preview Logic  (separated from final computation)
    # ==========================================================

    def _update_low_preview(self):
        """
        Called when σ_low slider changes.

        Updates ONLY the Image A preview with its Gaussian-blurred version.
        The hybrid result is NOT recomputed here — that only happens on
        Process button click.  This gives the user a live preview of the
        low-frequency component they are controlling.
        """
        if self.originalA is None or self.stack.currentWidget() != self.hybrid_page:
            return

        sigma = self.hybrid_sigma_low.value()
        self.hybrid_sigma_low_val.setText(str(sigma))

        # Compute low-pass component and store it (float32 for precision)
        self.low_component = cv2.GaussianBlur(
            self.originalA.astype(np.float32), (0, 0), sigma
        )

        # Display the blurred preview (converted to uint8 for rendering)
        display_low = np.clip(self.low_component, 0, 255).astype(np.uint8)
        self._set_scaled_pixmap(self.hybrid_label_a, display_low)

    def _update_high_preview(self):
        """
        Called when σ_high slider changes.

        Updates ONLY the Image B preview with its high-frequency component.
        The high-frequency component is computed as:

            high = originalB − GaussianBlur(originalB)

        Why zero-centered storage?
            Subtracting the blurred image yields values in ≈[-255, +255],
            centered around zero.  This raw representation preserves the
            true signed detail needed for correct hybrid addition later.

        Why +128 shift for preview?
            A zero-centered image would appear as a uniform gray blob on
            screen.  Shifting by +128 maps the zero-crossing to mid-gray,
            making both positive (bright edges) and negative (dark edges)
            visible to the user during tuning.
        """
        if self.originalB is None or self.stack.currentWidget() != self.hybrid_page:
            return

        sigma = self.hybrid_sigma_high.value()
        self.hybrid_sigma_high_val.setText(str(sigma))

        b_float = self.originalB.astype(np.float32)
        blurred_b = cv2.GaussianBlur(b_float, (0, 0), sigma)

        # Store the RAW zero-centered high component (float32)
        self.high_component = b_float - blurred_b

        # Preview: shift by +128 so zero-crossings appear as mid-gray
        high_display = self.high_component + 128.0
        high_display = cv2.normalize(high_display, None, 0, 255, cv2.NORM_MINMAX)
        self._set_scaled_pixmap(self.hybrid_label_b, high_display.astype(np.uint8))

    def _compute_and_display_hybrid(self):
        """
        Called ONLY when the user clicks 'Process Hybrid'.

        Combines the stored low and high components:
            hybrid = low_component + high_component

        Why normalization is necessary:
            The sum of a low-pass blur ([0, 255]) and a zero-centered
            high-pass ([≈-255, +255]) can range from roughly -255 to +510.
            cv2.normalize maps this full range back to [0, 255] so the
            result is displayable without clipping artefacts.
        """
        if self.low_component is None or self.high_component is None:
            return

        hybrid = self.low_component.astype(np.float32) + self.high_component.astype(np.float32)
        hybrid = cv2.normalize(hybrid, None, 0, 255, cv2.NORM_MINMAX)
        self.hybrid_result = hybrid.astype(np.uint8)

        self._set_scaled_pixmap(self.hybrid_label_result, self.hybrid_result)
        self.hybrid_result_ready.emit(self.hybrid_result)

    def display_hybrid(self, img_a: np.ndarray, img_b: np.ndarray):
        """
        Switch to the hybrid page, display both source images,
        and initialize internal components.

        The hybrid result is NOT computed here — the user must click
        Process to see the final blend.
        """
        self.originalA = img_a.copy()

        # Resize Image B to match Image A if needed
        h, w = self.originalA.shape[:2]
        self.originalB = cv2.resize(img_b, (w, h))

        # Initialize components to originals
        self.low_component = self.originalA.astype(np.float32)
        self.high_component = self.originalB.astype(np.float32) - cv2.GaussianBlur(
            self.originalB.astype(np.float32), (0, 0), self.hybrid_sigma_high.value()
        )
        self.hybrid_result = None

        self.stack.setCurrentWidget(self.hybrid_page)

        # Display originals in previews
        self._set_scaled_pixmap(self.hybrid_label_a, self.originalA)
        self._set_scaled_pixmap(self.hybrid_label_b, self.originalB)
        self.hybrid_label_result.clear()
        self.hybrid_label_result.setText("Click \"Process Hybrid\" to generate")

    def _create_grid_label(self, title):
        """Helper to create consistent labels for the 4-grid view."""
        container = QWidget()
        lay = QVBoxLayout(container)
        lay.setContentsMargins(0,0,0,0)

        title_lbl = QLabel(title)
        title_lbl.setStyleSheet("color: #007ACC; font-weight: bold;")
        img_lbl = QLabel()
        img_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        img_lbl.setObjectName("canvas_placeholder")

        img_lbl.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        img_lbl.setMinimumSize(1, 1)

        lay.addWidget(title_lbl)
        lay.addWidget(img_lbl, stretch=1)
        # Attach the image label to the container so we can access it later
        container.img_lbl = img_lbl
        return container

    # --- Core Functionality ---

    def load_image_dialog(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        if file_path:
            # Load with OpenCV. It loads as BGR by default.
            img = cv2.imread(file_path, cv2.IMREAD_COLOR)
            if img is not None:
                self.original_image = img.copy()
                self.current_image = img.copy()
                self.display_single_image(self.current_image)
                self.image_loaded.emit(self.current_image)

    def reset_image(self):
        if self.original_image is not None:
            self.current_image = self.original_image.copy()
            self.display_single_image(self.current_image)
            self.image_loaded.emit(self.current_image)

    # --- OpenCV to PyQt Conversion (The "Bridge") ---

    def _cv_to_pixmap(self, cv_img):
        """Safely converts OpenCV ndarray to PyQt QPixmap."""
        if cv_img is None: return QPixmap()

        # Grayscale Images
        if len(cv_img.shape) == 2:
            h, w = cv_img.shape
            bytes_per_line = w
            q_img = QImage(cv_img.data, w, h, bytes_per_line, QImage.Format.Format_Grayscale8)

        # Color Images (RGB/BGR)
        elif len(cv_img.shape) == 3:
            h, w, ch = cv_img.shape
            bytes_per_line = ch * w
            # OpenCV uses BGR, PyQt uses RGB. We must convert or the colors will look alien!
            rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            q_img = QImage(rgb_img.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)

        else:
            return QPixmap()

        return QPixmap.fromImage(q_img)

    def _set_scaled_pixmap(self, label, cv_img):
        """Converts and scales the image to fit the label without breaking the aspect ratio."""
        pixmap = self._cv_to_pixmap(cv_img)
        # Scale the pixmap to fit the label size while keeping aspect ratio
        scaled_pixmap = pixmap.scaled(label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        label.setPixmap(scaled_pixmap)

    # --- Layout Routers (Called by MainWindow) ---

    def display_single_image(self, cv_img):
        """Switches to the single view and displays the image."""
        self.stack.setCurrentWidget(self.single_page)
        self._set_scaled_pixmap(self.single_label, cv_img)

    def display_edge_grid(self, original, x_img, y_img, mag_img):
        """Switches to the 4-grid view and populates all four panes."""
        self.stack.setCurrentWidget(self.grid_page)
        self._set_scaled_pixmap(self.grid_orig.img_lbl, original)
        self._set_scaled_pixmap(self.grid_x.img_lbl, x_img)
        self._set_scaled_pixmap(self.grid_y.img_lbl, y_img)
        self._set_scaled_pixmap(self.grid_mag.img_lbl, mag_img)

    # Automatically resize the images if the user resizes the application window
    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.current_image is not None:
            if self.stack.currentWidget() == self.single_page:
                self.display_single_image(self.current_image)
        # Re-render hybrid page on resize
        if self.stack.currentWidget() == self.hybrid_page:
            # Re-render whichever components exist
            if self.low_component is not None:
                display_low = np.clip(self.low_component, 0, 255).astype(np.uint8)
                self._set_scaled_pixmap(self.hybrid_label_a, display_low)
            elif self.originalA is not None:
                self._set_scaled_pixmap(self.hybrid_label_a, self.originalA)
            if self.originalB is not None:
                self._set_scaled_pixmap(self.hybrid_label_b, self.originalB)
            if self.hybrid_result is not None:
                self._set_scaled_pixmap(self.hybrid_label_result, self.hybrid_result)
