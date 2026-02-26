import numpy as np
from PyQt6.QtWidgets import QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QMessageBox
from PyQt6.QtCore import Qt

from ui.sidebar_controls import SidebarControls
from ui.canvas_view import CanvasView
from ui.histogram_panel import HistogramPanel
from workers.thread_workers import ImageWorker

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("VisionLab Pro - Spatial & Frequency Processing")
        self.resize(1280, 720)

        # Application State
        self.base_image = None     # The unmodified image loaded from disk
        self.current_image = None  # The currently displayed image state
        self.worker = None         # Holds the background QThread

        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Instantiate the three main UI pillars
        self.sidebar = SidebarControls()
        self.canvas = CanvasView()
        self.histogram = HistogramPanel()

        # Build the Right Viewport (Canvas on top, Histogram on bottom)
        right_viewport = QWidget()
        right_layout = QVBoxLayout(right_viewport)
        right_layout.setContentsMargins(10, 10, 10, 10)
        right_layout.addWidget(self.canvas, stretch=1)
        right_layout.addWidget(self.histogram)

        main_layout.addWidget(self.sidebar)
        main_layout.addWidget(right_viewport, stretch=1)

    def _connect_signals(self):
        # 1. Listen for the recipe from the Sidebar
        self.sidebar.process_requested.connect(self.handle_pipeline_execution)
        
        # 2. Listen for a new image being loaded in the Canvas
        self.canvas.image_loaded.connect(self.on_image_loaded)

    # --- State Management ---

    def on_image_loaded(self, image: np.ndarray):
        """Called whenever the user loads a file or clicks 'Reset'."""
        self.base_image = image.copy()
        self.current_image = image.copy()
        self.histogram.update_plots(self.current_image)

    # --- Worker Thread Execution ---

    def handle_pipeline_execution(self, recipe: list):
        """Dispatches the recipe to the background worker."""
        if self.base_image is None:
            QMessageBox.warning(self, "No Image", "Please load an image first!")
            return

        # UX Enhancement: Lock the Process button to prevent spam-clicking
        self.sidebar.process_btn.setEnabled(False)
        self.sidebar.process_btn.setText("⏳ Processing...")

        # Spawn the worker with the unmodified BASE image
        self.worker = ImageWorker(self.base_image, recipe)
        self.worker.result_ready.connect(self.on_worker_finished)
        self.worker.error_occurred.connect(self.on_worker_error)
        self.worker.start()

    # --- Layout Routing (The Smart Controller) ---

    def on_worker_finished(self, response: dict):
        """Routes the worker's output dictionary to the correct canvas layout."""
        # Unlock the UI
        self.sidebar.process_btn.setEnabled(True)
        self.sidebar.process_btn.setText("▶ Process Image")

        action = response.get("action")
        data = response.get("data")

        # Python 3.10+ Pattern Matching
        match action:
            case "SobelEdge" | "PrewittEdge" | "RobertsEdge":
                if isinstance(data, dict):
                # These operations return a Multi-Buffer Dictionary
                    self.canvas.display_edge_grid(
                        original=self.base_image,
                        x_img=data["x"],
                        y_img=data["y"],
                        mag_img=data["magnitude"]
                    )
                    self.current_image = data["magnitude"]

            case _:
                # Standard operations ("None", "GaussianNoise", "CannyEdge") return ndarrays
                self.canvas.display_single_image(data)
                self.current_image = data

        # Update the analytics drawer
        self.histogram.update_plots(self.current_image)

    def on_worker_error(self, error_msg: str):
        """Handles backend math crashes gracefully without bringing down the app."""
        self.sidebar.process_btn.setEnabled(True)
        self.sidebar.process_btn.setText("▶ Process Image")
        QMessageBox.critical(self, "Processing Error", f"The pipeline encountered an error:\n{error_msg}")