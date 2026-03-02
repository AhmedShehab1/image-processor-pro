import cv2
import numpy as np
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                             QPushButton, QFileDialog, QStackedWidget, QGridLayout,
                             QSizePolicy)
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import Qt, pyqtSignal


class CanvasView(QWidget):
    """Standard-mode canvas for single-image and edge-grid views."""

    image_loaded = pyqtSignal(np.ndarray)
    undo_requested = pyqtSignal()
    redo_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_image = None
        self.original_image = None
        self._setup_ui()

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)

        # --- Top Toolbar ---
        toolbar = QHBoxLayout()
        self.load_btn = QPushButton("📁 Load Image")
        self.load_btn.clicked.connect(self.load_image_dialog)

        self.reset_btn = QPushButton("⏪ Reset to Original")
        self.reset_btn.clicked.connect(self.reset_image)

        self.undo_btn = QPushButton("↩ Undo")
        self.redo_btn = QPushButton("↪ Redo")
        self.undo_btn.setEnabled(False)
        self.redo_btn.setEnabled(False)

        self.undo_btn.clicked.connect(self.undo_requested.emit)
        self.redo_btn.clicked.connect(self.redo_requested.emit)

        toolbar.addWidget(self.load_btn)
        toolbar.addWidget(self.reset_btn)
        toolbar.addWidget(self.undo_btn)
        toolbar.addWidget(self.redo_btn)
        toolbar.addStretch()
        main_layout.addLayout(toolbar)

        # --- Stacked Widget for Layout Routing ---
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
        container.img_lbl = img_lbl
        return container

    # --- Core Functionality ---

    def load_image_dialog(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        if file_path:
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

    # --- OpenCV to PyQt Conversion ---

    def _cv_to_pixmap(self, cv_img):
        """Safely converts OpenCV ndarray to PyQt QPixmap."""
        if cv_img is None: return QPixmap()

        if len(cv_img.shape) == 2:
            h, w = cv_img.shape
            bytes_per_line = w
            q_img = QImage(cv_img.data, w, h, bytes_per_line, QImage.Format.Format_Grayscale8)
        elif len(cv_img.shape) == 3:
            h, w, ch = cv_img.shape
            bytes_per_line = ch * w
            rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            q_img = QImage(rgb_img.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        else:
            return QPixmap()

        return QPixmap.fromImage(q_img)

    def _set_scaled_pixmap(self, label, cv_img):
        """Converts and scales the image to fit the label without breaking the aspect ratio."""
        pixmap = self._cv_to_pixmap(cv_img)
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

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.current_image is not None:
            if self.stack.currentWidget() == self.single_page:
                self.display_single_image(self.current_image)
