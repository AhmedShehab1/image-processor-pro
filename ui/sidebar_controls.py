import cv2
import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QPushButton, QSlider, QComboBox, QLabel,
    QRadioButton, QButtonGroup, QToolBox, QHBoxLayout
)
from PyQt6.QtCore import Qt, pyqtSignal
from core.config_models import (
    NoiseConfig, SpatialConfig, EdgeConfig, FrequencyConfig,
    EnhancementConfig, ColorToGrayConfig
)

class SidebarControls(QWidget):
    process_requested = pyqtSignal(list)  # Emits the full recipe (list of steps)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(300)
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(10, 10, 10, 10)

        self.pipeline_state = {
            "noise": None,
            "spatial": None,
            "edge": None,
            "frequency": None,
            "enhancement": None,
            "color": None
        }

        # Main Accordion Widget
        self.toolbox = QToolBox()
        self.layout.addWidget(self.toolbox)

        # Build individual accordion sections
        self._build_noise_section()
        self._build_spatial_section()
        self._build_edge_section()
        self._build_frequency_section()
        self._build_enhancement_section()
        self._build_color_section()

        # Global Process Button
        self.process_btn = QPushButton("▶ Process Image")
        self.process_btn.setMinimumHeight(40)
        self.process_btn.setObjectName("process_btn")
        self.process_btn.clicked.connect(self._emit_process_signal)
        self.layout.addWidget(self.process_btn)

    # --- 1. Noise Injection ---
    def _build_noise_section(self):
        widget = QWidget()
        lay = QVBoxLayout(widget)

        lay.addWidget(QLabel("Noise Model:"))
        self.noise_dropdown = QComboBox()
        self.noise_dropdown.addItems(["Gaussian", "Uniform", "Salt & Pepper"])
        lay.addWidget(self.noise_dropdown)

        lay.addWidget(QLabel("Intensity / Probability:"))
        self.noise_slider = QSlider(Qt.Orientation.Horizontal)
        self.noise_slider.setRange(0, 100)
        self.noise_slider.setValue(25)
        lay.addWidget(self.noise_slider)

        self.noise_dropdown.currentIndexChanged.connect(self._update_noise_step)
        self.noise_slider.valueChanged.connect(self._update_noise_step)

        lay.addStretch()
        self.toolbox.addItem(widget, "1. Noise Injection")

    def _update_noise_step(self):
        self.pipeline_state["noise"] = NoiseConfig(
                    model=self.noise_dropdown.currentText(),
                    intensity=self.noise_slider.value()
        )

    # --- 2. Spatial Adjustments ---
    def _build_spatial_section(self):
        widget = QWidget()
        lay = QVBoxLayout(widget)

        self.spatial_group = QButtonGroup(widget)
        for i, name in enumerate(["Average", "Gaussian", "Median"]):
            btn = QRadioButton(name)
            if i == 0: btn.setChecked(True)
            self.spatial_group.addButton(btn, i)
            lay.addWidget(btn)
            btn.toggled.connect(self._update_spatial_step)

        lay.addWidget(QLabel("Kernel Size (3, 5, 7):"))
        self.kernel_slider = QSlider(Qt.Orientation.Horizontal)
        self.kernel_slider.setRange(1, 3)
        self.kernel_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.kernel_slider.setTickInterval(1)
        self.kernel_slider.setValue(1)
        self.kernel_slider.valueChanged.connect(self._update_spatial_step)
        lay.addWidget(self.kernel_slider)

        lay.addWidget(QLabel("Sigma (For Gaussian):"))
        self.sigma_slider = QSlider(Qt.Orientation.Horizontal)
        self.sigma_slider.setRange(1, 50)
        self.sigma_slider.setValue(12)
        self.sigma_slider.valueChanged.connect(self._update_spatial_step)
        lay.addWidget(self.sigma_slider)

        lay.addStretch()
        self.toolbox.addItem(widget, "2. Spatial Filters")

    def _update_spatial_step(self):
        self.pipeline_state["spatial"] = SpatialConfig(
                    filter_type=self.spatial_group.checkedButton().text(),
                    kernel_size=(self.kernel_slider.value() * 2) + 1,
                    sigma=self.sigma_slider.value() / 10.0
                )

    # --- 3. Edge Detection ---
    def _build_edge_section(self):
        widget = QWidget()
        lay = QVBoxLayout(widget)

        self.edge_group = QButtonGroup(widget)
        for i, name in enumerate(["Sobel", "Roberts", "Prewitt", "Canny"]):
            btn = QRadioButton(name)
            if i == 0: btn.setChecked(True)
            self.edge_group.addButton(btn, i)
            lay.addWidget(btn)
            btn.toggled.connect(self._update_edge_step)
            btn.toggled.connect(self._toggle_canny_sliders)

        self.canny_widget = QWidget()
        canny_lay = QVBoxLayout(self.canny_widget)
        canny_lay.setContentsMargins(0,0,0,0)

        canny_lay.addWidget(QLabel("Canny Min Threshold:"))
        self.canny_min = QSlider(Qt.Orientation.Horizontal)
        self.canny_min.setRange(0, 255)
        self.canny_min.setValue(100)
        self.canny_min.valueChanged.connect(self._update_edge_step)
        canny_lay.addWidget(self.canny_min)

        canny_lay.addWidget(QLabel("Canny Max Threshold:"))
        self.canny_max = QSlider(Qt.Orientation.Horizontal)
        self.canny_max.setRange(0, 255)
        self.canny_max.setValue(200)
        self.canny_max.valueChanged.connect(self._update_edge_step)
        canny_lay.addWidget(self.canny_max)

        self.canny_widget.setVisible(False)
        lay.addWidget(self.canny_widget)

        lay.addStretch()
        self.toolbox.addItem(widget, "3. Edge Detection")

    def _toggle_canny_sliders(self):
        is_canny = self.edge_group.checkedId() == 3
        self.canny_widget.setVisible(is_canny)

    def _update_edge_step(self):
        operator = self.edge_group.checkedButton().text()

        if operator == "Canny":
            self.pipeline_state["edge"] = EdgeConfig(
                operator=operator,
                canny_min=self.canny_min.value(),
                canny_max=self.canny_max.value()
            )
        else:
            self.pipeline_state["edge"] = EdgeConfig(
                operator=operator
        )

    # --- 4. Frequency Domain ---
    def _build_frequency_section(self):
        widget = QWidget()
        lay = QVBoxLayout(widget)

        lay.addWidget(QLabel("Filter Type:"))
        self.freq_dropdown = QComboBox()
        self.freq_dropdown.addItems(["Low-Pass", "High-Pass"])
        self.freq_dropdown.currentIndexChanged.connect(self._update_frequency_step)
        lay.addWidget(self.freq_dropdown)

        lay.addWidget(QLabel("Cutoff Radius (D0):"))
        self.freq_slider = QSlider(Qt.Orientation.Horizontal)
        self.freq_slider.setRange(1, 200)
        self.freq_slider.setValue(50)
        self.freq_slider.valueChanged.connect(self._update_frequency_step)
        lay.addWidget(self.freq_slider)

        lay.addStretch()
        self.toolbox.addItem(widget, "4. Frequency Domain")

    def _update_frequency_step(self):
        filter_type = self.freq_dropdown.currentText()
        cutoff_radius = self.freq_slider.value()
        self.pipeline_state["frequency"] = FrequencyConfig(
            filter_type=filter_type,
            cutoff_radius=cutoff_radius
        )

    # --- 5. Global Enhancements ---
    def _build_enhancement_section(self):
        widget = QWidget()
        lay = QVBoxLayout(widget)

        self.equalize_btn = QPushButton("Equalize Histogram")
        self.normalize_btn = QPushButton("Normalize Image")
        self.equalize_btn.clicked.connect(self._add_equalize_step)
        self.normalize_btn.clicked.connect(self._add_normalize_step)

        lay.addWidget(self.equalize_btn)
        lay.addWidget(self.normalize_btn)

        lay.addStretch()
        self.toolbox.addItem(widget, "5. Enhancements")

    def _add_equalize_step(self):
            self.pipeline_state["enhancement"] = EnhancementConfig(action_type="Equalize")

    def _add_normalize_step(self):
        self.pipeline_state["enhancement"] = EnhancementConfig(action_type="Normalize")

    # --- 6. Color Conversion ---
    def _build_color_section(self):
        widget = QWidget()
        lay = QVBoxLayout(widget)

        self.gray_btn = QPushButton("Convert to Grayscale")
        self.gray_btn.clicked.connect(self._add_gray_step)
        lay.addWidget(self.gray_btn)

        lay.addStretch()
        self.toolbox.addItem(widget, "6. Color Conversion")

    def _add_gray_step(self):
        self.pipeline_state["color"] = ColorToGrayConfig(method="Manual")

    # --- Communication Logic ---
    def _emit_process_signal(self):
        recipe = []
        processing_order = ["color", "enhancement", "noise", "spatial", "frequency", "edge"]

        for stage in processing_order:
            config = self.pipeline_state.get(stage)
            if config is not None:
                recipe.append(config)

        self.process_requested.emit(recipe)
