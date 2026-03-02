import numpy as np
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QMessageBox, QTabWidget
)
from PyQt6.QtCore import Qt

from ui.sidebar_controls import SidebarControls
from ui.canvas_view import CanvasView
from ui.histogram_panel import HistogramPanel
from ui.hybrid_mode import HybridModeWidget
from workers.thread_workers import ImageWorker
from config import AppConfig

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("VisionLab Pro - Spatial & Frequency Processing")
        self.resize(1280, 720)

        # Application State
        self.base_image = None     # The unmodified image loaded from disk
        self.current_image = None  # The currently displayed image state
        self.current_multi_buffer = None  # Holds multi-buffer outputs for edge detection, etc.
        self.last_action = None   # A string hint of the last operation performed (for undo/redo context)
        self.worker = None         # Holds the background QThread

        self.history_stack = []
        self.redo_stack = []
        self.MAX_HISTORY = AppConfig.MAX_HISTORY

        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        root_layout = QVBoxLayout(central_widget)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        # ---- Top-Level Tab Widget ----
        self.tabs = QTabWidget()
        self.tabs.setObjectName("main_tabs")
        root_layout.addWidget(self.tabs)

        # --- Tab 1: Standard Mode ---
        standard_tab = QWidget()
        standard_layout = QHBoxLayout(standard_tab)
        standard_layout.setContentsMargins(0, 0, 0, 0)
        standard_layout.setSpacing(0)

        self.sidebar = SidebarControls()
        self.canvas = CanvasView()
        self.histogram = HistogramPanel()

        # Right Viewport (Canvas + Histogram)
        right_viewport = QWidget()
        right_layout = QVBoxLayout(right_viewport)
        right_layout.setContentsMargins(10, 10, 10, 10)
        right_layout.addWidget(self.canvas, stretch=1)
        right_layout.addWidget(self.histogram)

        standard_layout.addWidget(self.sidebar)
        standard_layout.addWidget(right_viewport, stretch=1)

        self.tabs.addTab(standard_tab, "📐 Standard Mode")

        # --- Tab 2: Hybrid Mode ---
        self.hybrid_widget = HybridModeWidget()
        self.tabs.addTab(self.hybrid_widget, "🔬 Hybrid Mode")

    def _connect_signals(self):
        # 1. Listen for the recipe from the Sidebar
        self.sidebar.process_requested.connect(self.handle_pipeline_execution)

        # 2. Listen for a new image being loaded in the Canvas
        self.canvas.image_loaded.connect(self.on_image_loaded)

        self.canvas.undo_requested.connect(self.perform_undo)
        self.canvas.redo_requested.connect(self.perform_redo)

        # 3. Hybrid mode → histogram updates
        self.hybrid_widget.hybrid_computed.connect(self._on_hybrid_computed)

    # --- State Management ---

    def on_image_loaded(self, image: np.ndarray):
        """Called whenever the user loads a file or clicks 'Reset'."""
        self.base_image = image.copy()
        self.current_image = image.copy()
        self.current_multi_buffer = None  # Clear any previous multi-buffer state

        self.history_stack.clear()
        self.redo_stack.clear()
        self._update_undo_redo_buttons()

        self.histogram.update_plots(self.current_image)

    def _on_hybrid_computed(self, result: np.ndarray):
        """Called when the hybrid mode produces a new result."""
        self.histogram.update_plots(result)

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
        if self.current_image is not None:
            self.history_stack.append({
                "image": self.current_image.copy(),
                "multi_buffer": self.current_multi_buffer.copy() if self.current_multi_buffer else None,
                "action": self.last_action
            })

            if len(self.history_stack) > self.MAX_HISTORY:
                self.history_stack.pop(0)

            self.redo_stack.clear()  # Clear the redo stack on new action

        action = response.get("action", "Unknown")
        data = response.get("data")
        self.last_action = action  # Store the last action for undo/redo context

        # --- Route by Data Structure, NOT by Magic Strings ---

        # Scenario A: We received a multi-buffer dictionary
        if isinstance(data, dict):

            self.current_image = data["magnitude"] if "magnitude" in data else self.current_image
            self.current_multi_buffer = data

        # Scenario B: We received a standard image array
        elif isinstance(data, np.ndarray):
            self.current_image = data
            self.current_multi_buffer = None

        # Scenario C: Something went terribly wrong
        else:
            print(f"Error: Unrecognized data type {type(data)} from {action}")


        self._render_current_state()
        self.sidebar.process_btn.setEnabled(True)
        self.sidebar.process_btn.setText("▶ Process Image")
        # Update the analytics drawer
        if self.current_image is not None:
            self.histogram.update_plots(self.current_image)

    def perform_undo(self):
        if not self.history_stack: return

        # 1. Save CURRENT state to Redo stack
        self.redo_stack.append({
            "image": self.current_image.copy(),
            "multi_buffer": self.current_multi_buffer.copy() if self.current_multi_buffer is not None else None,
            "action": self.last_action
        })

        # 2. Pop PREVIOUS state from History
        prev_state = self.history_stack.pop()
        self.current_image = prev_state["image"]
        self.current_multi_buffer = prev_state["multi_buffer"]
        self.last_action = prev_state["action"]

        self._render_current_state()

    def perform_redo(self):
        if not self.redo_stack: return

        # 1. Save CURRENT state to History stack
        self.history_stack.append({
            "image": self.current_image.copy(),
            "multi_buffer": self.current_multi_buffer,
            "action": self.last_action
        })

        # 2. Pop NEXT state from Redo
        next_state = self.redo_stack.pop()
        self.current_image = next_state["image"]
        self.current_multi_buffer = next_state["multi_buffer"]
        self.last_action = next_state["action"]

        self._render_current_state()

    def _render_current_state(self):
        """Routes the current memory state to the correct Canvas view."""
        if self.current_multi_buffer is not None:
            data = self.current_multi_buffer
            if {"x", "y", "magnitude"}.issubset(data.keys()):
                self.canvas.display_edge_grid(
                    original=self.base_image,
                    x_img=data["x"],
                    y_img=data["y"],
                    mag_img=data["magnitude"]
                )
        else:
            self.canvas.display_single_image(self.current_image)

        self.histogram.update_plots(self.current_image)
        self._update_undo_redo_buttons()

    def _update_undo_redo_buttons(self):
        """Dynamically disables buttons if there is no history to prevent crashes."""
        self.canvas.undo_btn.setEnabled(len(self.history_stack) > 0)
        self.canvas.redo_btn.setEnabled(len(self.redo_stack) > 0)

    def on_worker_error(self, error_msg: str):
        """Handles backend math crashes gracefully without bringing down the app."""
        self.sidebar.process_btn.setEnabled(True)
        self.sidebar.process_btn.setText("▶ Process Image")
        QMessageBox.critical(self, "Processing Error", f"The pipeline encountered an error:\n{error_msg}")