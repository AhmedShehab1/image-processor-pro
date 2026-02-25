# operation_factory.py
from core.config_models import *
from core.operations import * 

def build_operation(config: OperationConfig) -> ImageOperation:
    """Translates UI Data Models into OpenCV execution classes."""
    
    match config:
        # --- Noise Routing ---
        case NoiseConfig(model="Gaussian", intensity=i):
            return GaussianNoise(intensity=i)
        
        case NoiseConfig(model="Uniform", intensity=i):
            return UniformNoise(intensity=i)
            
        case NoiseConfig(model="Salt & Pepper", intensity=i):
            # Scale UI slider (0-100) to a probability float (0.0-1.0)
            return SaltPepperNoise(probability=i / 100.0)

        # --- Spatial Filter Routing ---
        case SpatialConfig(filter_type="Average", kernel_size=k):
            return AverageFilter(kernel_size=k)
            
        case SpatialConfig(filter_type="Gaussian", kernel_size=k, sigma=s):
            return GaussianFilter(kernel_size=k, sigma=s)
            
        case SpatialConfig(filter_type="Median", kernel_size=k):
            return MedianFilter(kernel_size=k)

        # --- Edge Detection Routing ---
        case EdgeConfig(operator="Sobel"):
            return SobelEdge()
            
        case EdgeConfig(operator="Prewitt"):
            return PrewittEdge()
            
        case EdgeConfig(operator="Roberts"):
            return RobertsEdge()
            
        case EdgeConfig(operator="Canny", canny_min=min_val, canny_max=max_val):
            return CannyEdge(threshold1=min_val, threshold2=max_val)

        # --- Frequency Domain Routing ---
        case FrequencyConfig(filter_type="Low-Pass", cutoff_radius=r):
            return LowPassFilter(cutoff=r)
            
        case FrequencyConfig(filter_type="High-Pass", cutoff_radius=r):
            return HighPassFilter(cutoff=r)

        # --- Enhancement Routing ---
        case EnhancementConfig(action_type="Equalize"):
            return EqualizeHistogram()
            
        case EnhancementConfig(action_type="Normalize"):
            return NormalizeImage()

        case _:
            raise ValueError(f"Factory cannot map unknown config: {type(config)}")
