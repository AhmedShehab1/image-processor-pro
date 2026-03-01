from dataclasses import dataclass, field
from typing import Optional
import numpy as np

@dataclass
class OperationConfig:
    """Base class for all UI operation requests."""
    pass

@dataclass
class NoiseConfig(OperationConfig):
    model: str           # "Gaussian", "Uniform", "Salt & Pepper"
    intensity: float     

@dataclass
class SpatialConfig(OperationConfig):
    filter_type: str     # "Average", "Gaussian", "Median"
    kernel_size: int     
    sigma: float         

@dataclass
class EdgeConfig(OperationConfig):
    operator: str        # "Sobel", "Roberts", "Prewitt", "Canny"
    canny_min: int = 100
    canny_max: int = 200

@dataclass
class FrequencyConfig(OperationConfig):
    filter_type: str     # "Low-Pass", "High-Pass"
    cutoff_radius: int


@dataclass
class EnhancementConfig(OperationConfig):
    action_type: str  # "Equalize" or "Normalize"

@dataclass
class ColorToGrayConfig(OperationConfig):
    method: str  # "Manual"

@dataclass
class HybridConfig(OperationConfig):
    sigma_low: float
    sigma_high: float
    second_image: np.ndarray = field(default=None, repr=False)