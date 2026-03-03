"""
ClickableImageLabel — A QLabel that opens a file dialog when clicked.

Replaces traditional "Load Image" buttons with a clickable image area,
providing a more intuitive UX where the user clicks the empty image slot
to load a file.
"""

import cv2
import numpy as np
from PyQt6.QtWidgets import QLabel, QFileDialog, QSizePolicy
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import Qt, pyqtSignal


class ClickableImageLabel(QLabel):
    """
    A QLabel that acts as a clickable image loader.

    When empty, displays a placeholder prompt.  On click, opens a file
    dialog, loads the selected image with OpenCV, and emits the loaded
    ndarray via the ``image_loaded`` signal.

    Signals:
        image_loaded(np.ndarray): BGR uint8 image that was loaded.
    """

    image_loaded = pyqtSignal(np.ndarray)

    def __init__(self, placeholder_text: str = "Click to load image", parent=None):
        super().__init__(parent)
        self.setText(placeholder_text)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setObjectName("canvas_placeholder")
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.setMinimumSize(1, 1)
        self.setStyleSheet("border: 2px dashed #3F3F46; border-radius: 6px;")
        self.setCursor(Qt.CursorShape.PointingHandCursor)

        self._cv_image = None  # Stores the loaded OpenCV image

    def mousePressEvent(self, event):
        """Open a file dialog on click and load the selected image."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Image", "", "Images (*.png *.jpg *.jpeg *.bmp)"
        )
        if file_path:
            img = cv2.imread(file_path, cv2.IMREAD_COLOR)
            if img is not None:
                self._cv_image = img
                self._display(img)
                self.image_loaded.emit(img)

    def set_image(self, cv_img: np.ndarray):
        """Programmatically set an image without the file dialog."""
        self._cv_image = cv_img
        self._display(cv_img)

    def _display(self, cv_img: np.ndarray):
        """Convert an OpenCV BGR image to a scaled QPixmap and display it."""
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
            self.size(), Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.setPixmap(scaled)

    def resizeEvent(self, event):
        """Re-scale the image when the widget is resized."""
        super().resizeEvent(event)
        if self._cv_image is not None:
            self._display(self._cv_image)
