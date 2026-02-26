import numpy as np
from PyQt6.QtWidgets import QWidget, QVBoxLayout
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

class HistogramPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(220)  # Keep the drawer compact so the image stays large
        
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)

        # 1. Setup Matplotlib Figure with Graphite Dark Theme
        self.fig = Figure(facecolor='#1E1E1E')
        self.canvas = FigureCanvas(self.fig)
        self.layout.addWidget(self.canvas)

        # 2. Setup Primary Axis (For the Histogram bars)
        self.ax = self.fig.add_subplot(111)
        
        # 3. Setup Secondary Axis (For the CDF line curve, sharing the X-axis)
        self.ax_cdf = self.ax.twinx()
        
        self._style_axes()

    def _style_axes(self):
        """Applies the professional dark theme styling to the graph axes."""
        self.ax.set_facecolor('#1E1E1E')
        self.ax.tick_params(colors='#D4D4D4', labelsize=8)
        self.ax.spines['bottom'].set_color('#555555')
        self.ax.spines['top'].set_color('none')
        self.ax.spines['left'].set_color('none')
        self.ax.spines['right'].set_color('none')
        self.ax.set_yticks([]) # Hide Y ticks for a cleaner look
        
        self.ax_cdf.tick_params(colors='#007ACC', labelsize=8)
        self.ax_cdf.spines['top'].set_color('none')
        self.ax_cdf.spines['left'].set_color('none')
        self.ax_cdf.spines['right'].set_color('none')
        self.ax_cdf.spines['bottom'].set_color('#555555')
        
        self.fig.tight_layout(pad=1.0)

    def update_plots(self, image: np.ndarray):
        """Called by the MainWindow whenever the image state changes."""
        # Clear the old graphs
        self.ax.clear()
        self.ax_cdf.clear()
        self._style_axes()

        if image is None:
            self.canvas.draw()
            return

        # Route the plotting logic based on the image's color space
        if len(image.shape) == 2:
            self._plot_grayscale(image)
        elif len(image.shape) == 3:
            self._plot_rgb(image)

        # Lock the X-axis to standard pixel intensity range (0-255)
        self.ax.set_xlim([0, 255])
        # Lock the Y-axis for CDF between 0.0 and 1.05 (for slight padding)
        self.ax_cdf.set_ylim([0, 1.05])
        
        self.canvas.draw()

    def _plot_grayscale(self, image):
        """Plots a single gray histogram and its CDF."""
        # Calculate frequencies
        hist, bins = np.histogram(image.flatten(), 256, [0, 256])
        
        # Calculate normalized CDF
        cdf = hist.cumsum()
        cdf_normalized = cdf / cdf.max() if cdf.max() > 0 else cdf

        # Plot Histogram (Filled area)
        self.ax.fill_between(bins[:-1], hist, color='#888888', alpha=0.7)
        
        # Plot CDF (Solid blue line)
        self.ax_cdf.plot(cdf_normalized, color='#007ACC', linewidth=2, label="CDF")
        self.ax_cdf.legend(loc='upper left', frameon=False, labelcolor='#D4D4D4')

    def _plot_rgb(self, image):
        """Plots three separate histograms (R, G, B) and their CDFs."""
        # OpenCV uses BGR order, so index 0=Blue, 1=Green, 2=Red
        colors = ('#4287f5', '#42f563', '#f54242') 
        
        for i, col in enumerate(colors):
            # Calculate frequencies per channel
            hist, bins = np.histogram(image[:, :, i].flatten(), 256, [0, 256])
            
            # Calculate normalized CDF
            cdf = hist.cumsum()
            cdf_normalized = cdf / cdf.max() if cdf.max() > 0 else cdf

            # Plot Histogram (Line plot to avoid overlapping solid blocks)
            self.ax.plot(bins[:-1], hist, color=col, alpha=0.8, linewidth=1.5)
            
            # Plot CDF (Dashed line)
            self.ax_cdf.plot(cdf_normalized, color=col, linestyle='--', alpha=0.6)
