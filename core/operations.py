import cv2
import numpy as np
import scipy.signal
import scipy.ndimage
from abc import ABC, abstractmethod

# ==========================================
# 0) Base Strategy & Utility (The "Contract")
# ==========================================
class ImageOperation(ABC):
    """
    Base class for all image operations. 
    Strict Contract: Must accept an ndarray and return an ndarray.
    """
    @abstractmethod
    def apply(self, image: np.ndarray) -> np.ndarray:
        pass

class MultiOutputOperation(ImageOperation):
    """
    Extension for operations that generate multiple intermediate buffers 
    (like X and Y gradients). UI can safely ask for the extended dict.
    """
    @abstractmethod
    def apply_extended(self, image: np.ndarray) -> dict[str, np.ndarray]:
        pass

    def apply(self, image: np.ndarray) -> np.ndarray:
        """Satisfies the base ImageOperation contract by returning the primary result."""
        results = self.apply_extended(image)
        return results.get("magnitude", image)

def _apply_per_channel(func, image: np.ndarray, *args, **kwargs) -> np.ndarray:
    """Helper function to prevent convolutions from crashing on RGB (3D) images."""
    if len(image.shape) == 3: # If RGB
        channels = cv2.split(image)
        processed_channels = [func(c, *args, **kwargs) for c in channels]
        return cv2.merge(processed_channels)
    return func(image, *args, **kwargs) # If Grayscale


# ==========================================
# 1) Noise Generators
# ==========================================
class GaussianNoise(ImageOperation):
    def __init__(self, intensity=25):
        self.intensity = intensity

    def apply(self, image: np.ndarray) -> np.ndarray:
        img = image.copy().astype(np.float32)
        noise = np.random.normal(0, self.intensity, img.shape)
        return np.clip(img + noise, 0, 255).astype(np.uint8)

class UniformNoise(ImageOperation):
    def __init__(self, intensity=25):
        self.intensity = intensity

    def apply(self, image: np.ndarray) -> np.ndarray:
        img = image.copy().astype(np.float32)
        noise = np.random.uniform(-self.intensity, self.intensity, img.shape)
        return np.clip(img + noise, 0, 255).astype(np.uint8)

class SaltPepperNoise(ImageOperation):
    def __init__(self, probability=0.05):
        self.probability = probability

    def apply(self, image: np.ndarray) -> np.ndarray:
        noisy = image.copy()
        mask = np.random.random(noisy.shape)
        noisy[mask < (self.probability / 2)] = 0
        noisy[mask > 1 - (self.probability / 2)] = 255
        return noisy


# ==========================================
# 2) Spatial Filters (Noise Reduction)
# ==========================================
class AverageFilter(ImageOperation):
    def __init__(self, kernel_size=3):
        self.kernel_size = kernel_size

    def _process(self, single_channel):
        k = self.kernel_size
        kernel = np.ones((k, k), np.float32) / (k * k)
        return scipy.signal.convolve2d(single_channel.astype(np.float32), kernel, mode='same')

    def apply(self, image: np.ndarray) -> np.ndarray:
        res = _apply_per_channel(self._process, image)
        return np.clip(res, 0, 255).astype(np.uint8)

class GaussianFilter(ImageOperation):
    def __init__(self, kernel_size=3, sigma=1.0):
        self.kernel_size = kernel_size
        self.sigma = sigma

    def _process(self, single_channel):
        k = self.kernel_size
        ax = np.linspace(-(k // 2), k // 2, k)
        gauss = np.exp(-0.5 * (ax / self.sigma)**2)
        kernel = np.outer(gauss, gauss)
        kernel /= kernel.sum()
        return scipy.signal.convolve2d(single_channel.astype(np.float32), kernel, mode='same')

    def apply(self, image: np.ndarray) -> np.ndarray:
        res = _apply_per_channel(self._process, image)
        return np.clip(res, 0, 255).astype(np.uint8)

class MedianFilter(ImageOperation):
    def __init__(self, kernel_size=3):
        self.kernel_size = kernel_size

    def _process(self, single_channel):
        return scipy.ndimage.median_filter(single_channel, size=self.kernel_size)

    def apply(self, image: np.ndarray) -> np.ndarray:
        res = _apply_per_channel(self._process, image)
        return res.astype(np.uint8)


# ==========================================
# 3) Edge Detection 
# ==========================================
class ScratchEdgeDetector(MultiOutputOperation):
    """Base class for Sobel, Prewitt, and Roberts."""
    def __init__(self, kx, ky):
        self.kx = kx
        self.ky = ky

    def apply_extended(self, image: np.ndarray) -> dict[str, np.ndarray]:
        # Convert to grayscale if RGB before edge detection
        if len(image.shape) == 3:
            img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            img_gray = image

        img_float = img_gray.astype(float)
        gx = scipy.ndimage.convolve(img_float, self.kx)
        gy = scipy.ndimage.convolve(img_float, self.ky)
        mag = np.sqrt(gx**2 + gy**2)

        return {
            "x": np.clip(np.abs(gx), 0, 255).astype(np.uint8),
            "y": np.clip(np.abs(gy), 0, 255).astype(np.uint8),
            "magnitude": np.clip(mag, 0, 255).astype(np.uint8)
        }

class SobelEdge(ScratchEdgeDetector):
    def __init__(self):
        kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        super().__init__(kx, ky)

class PrewittEdge(ScratchEdgeDetector):
    def __init__(self):
        kx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
        ky = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
        super().__init__(kx, ky)

class RobertsEdge(ScratchEdgeDetector):
    def __init__(self):
        kx = np.array([[1, 0], [0, -1]])
        ky = np.array([[0, 1], [-1, 0]])
        super().__init__(kx, ky)

class CannyEdge(ImageOperation):
    def __init__(self, threshold1=100, threshold2=200):
        self.t1 = threshold1
        self.t2 = threshold2

    def apply(self, image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 3:
            img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            img_gray = image
        
        edges = cv2.Canny(img_gray, self.t1, self.t2)
        # Canny doesn't produce separate X/Y buffers natively, so it just returns the array
        return edges


# ==========================================
# 4) Frequency Domain Filters
# ==========================================
class BaseFrequencyFilter(ImageOperation):
    """Base class handling the complex FFT math."""
    def __init__(self, cutoff: int):
        self.cutoff = cutoff

    @abstractmethod
    def _create_mask(self, rows: int, cols: int) -> np.ndarray:
        pass

    def _process(self, single_channel: np.ndarray) -> np.ndarray:
        rows, cols = single_channel.shape
        crow, ccol = rows // 2, cols // 2

        # 1. Forward FFT & Shift origin to center
        f_transform = np.fft.fft2(single_channel)
        f_shift = np.fft.fftshift(f_transform)

        # 2. Get the mask from the subclass
        mask = self._create_mask(rows, cols)

        # 3. Apply Mask in frequency domain
        f_shift_filtered = f_shift * mask

        # 4. Inverse Shift & Inverse FFT
        f_ishift = np.fft.ifftshift(f_shift_filtered)
        img_back = np.fft.ifft2(f_ishift)
        
        # Take the absolute value to handle complex numbers
        return np.abs(img_back)

    def apply(self, image: np.ndarray) -> np.ndarray:
        res = _apply_per_channel(self._process, image)
        return np.clip(res, 0, 255).astype(np.uint8)


class LowPassFilter(BaseFrequencyFilter):
    def _create_mask(self, rows: int, cols: int) -> np.ndarray:
        crow, ccol = rows // 2, cols // 2
        # Create a boolean mask where True is inside the circle (radius = cutoff)
        y, x = np.ogrid[-crow:rows-crow, -ccol:cols-ccol]
        mask = x*x + y*y <= self.cutoff**2
        return mask


class HighPassFilter(BaseFrequencyFilter):
    def _create_mask(self, rows: int, cols: int) -> np.ndarray:
        crow, ccol = rows // 2, cols // 2
        # Create a boolean mask where True is OUTSIDE the circle
        y, x = np.ogrid[-crow:rows-crow, -ccol:cols-ccol]
        mask = x*x + y*y > self.cutoff**2
        return mask


# ==========================================
# 5) Global Enhancements
# ==========================================
class EqualizeHistogram(ImageOperation):
    def apply(self, image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 3:
            # For color images, we ONLY equalize the Luminance (Brightness) channel
            # Otherwise, equalizing RGB independently destroys the image colors.
            ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
            ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
            return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
        else:
            return cv2.equalizeHist(image)

class NormalizeImage(ImageOperation):
    def apply(self, image: np.ndarray) -> np.ndarray:
        # Stretches the pixel values to span the full 0-255 range
        return cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)


# ==========================================
# 6) Color Conversion
# ==========================================
class ManualGrayscale(ImageOperation):
    """
    Converts a color (BGR) image to grayscale using the ITU-R BT.601 luma formula:

        Gray = 0.299 * R + 0.587 * G + 0.114 * B

    Why manual formula instead of cv2.cvtColor?
        This implementation is required to demonstrate the mathematical basis of
        luminance-weighted grayscale conversion. The weights reflect human visual
        perception — green contributes most because the eye is most sensitive to it.

    Why CDF matters for histogram equalization:
        The CDF of the grayscale histogram acts as a mapping function: it remaps
        each intensity level to a new value that spreads pixel frequencies uniformly
        across the [0, 255] range, thereby improving contrast.
    """

    def apply(self, image: np.ndarray) -> np.ndarray:
        """
        Apply manual grayscale conversion.

        Args:
            image: Input image as a NumPy array (BGR 3-channel or 2D grayscale).

        Returns:
            2D uint8 grayscale image. Input is never mutated.
        """
        # Already grayscale — return a copy to guarantee no mutation
        if image.ndim == 2:
            return image.copy()

        # Extract BGR channels (OpenCV convention: index 0=B, 1=G, 2=R)
        b = image[:, :, 0].astype(np.float64)
        g = image[:, :, 1].astype(np.float64)
        r = image[:, :, 2].astype(np.float64)

        # ITU-R BT.601 luma formula — fully vectorized, no pixel loops
        gray = 0.299 * r + 0.587 * g + 0.114 * b

        return gray.astype(np.uint8)

