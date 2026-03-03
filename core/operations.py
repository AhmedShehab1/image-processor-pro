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


def center_crop_and_resize(img: np.ndarray, size: int = 512) -> np.ndarray:
    """
    Normalise an image to a square canvas of `size x size` pixels.

    Two-step process
    ----------------
    1. Centre-crop to the largest square that fits inside the original frame.
       This removes the black/letterbox bars that appear when two images with
       different aspect ratios are blended, and ensures both images contribute
       symmetrically in the frequency domain (no aspect-ratio distortion).

    2. Resize the square crop to `size x size` with bilinear interpolation.
       A fixed output resolution means the Gaussian sigma values computed by
       the adaptive formula are always in the same pixel-space, so a sigma
       that gives the right cutoff on a 512x512 image is equally valid for
       every image regardless of its original resolution.

    Parameters
    ----------
    img  : np.ndarray   Input image (grayscale 2-D or colour 3-D, any dtype).
    size : int          Side length of the output square (default 512).

    Returns
    -------
    np.ndarray  Square image with shape (size, size) or (size, size, C).
    """
    h, w = img.shape[:2]

    # 1) Crop the largest centred square
    min_dim = min(h, w)
    start_x = w // 2 - min_dim // 2
    start_y = h // 2 - min_dim // 2
    cropped = img[start_y : start_y + min_dim, start_x : start_x + min_dim]

    # 2) Resize to the target square resolution
    resized = cv2.resize(cropped, (size, size), interpolation=cv2.INTER_LINEAR)
    return resized


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

        f_transform = np.fft.fft2(single_channel)
        f_shift = np.fft.fftshift(f_transform)

        mask = self._create_mask(rows, cols)

        f_shift_filtered = f_shift * mask

        f_ishift = np.fft.ifftshift(f_shift_filtered)
        img_back = np.fft.ifft2(f_ishift)
        
        return np.abs(img_back)

    def apply(self, image: np.ndarray) -> np.ndarray:
        res = _apply_per_channel(self._process, image)
        return np.clip(res, 0, 255).astype(np.uint8)


class LowPassFilter(BaseFrequencyFilter):
    def _create_mask(self, rows: int, cols: int) -> np.ndarray:
        crow, ccol = rows // 2, cols // 2
        y, x = np.ogrid[-crow:rows-crow, -ccol:cols-ccol]
        mask = x*x + y*y <= self.cutoff**2
        return mask


class HighPassFilter(BaseFrequencyFilter):
    def _create_mask(self, rows: int, cols: int) -> np.ndarray:
        crow, ccol = rows // 2, cols // 2
        y, x = np.ogrid[-crow:rows-crow, -ccol:cols-ccol]
        mask = x*x + y*y > self.cutoff**2
        return mask


class HybridImage(ImageOperation):
    """
    Creates a hybrid image by combining:
      - The LOW frequencies  of `image_low`  -> seen when viewed from a distance
      - The HIGH frequencies of `image_high` -> seen when viewed up close

    Theory (Oliva et al. / lecture):
        Hybrid = LPF(image_low) + HPF(image_high)

    Correctness guarantees
    ----------------------
    FIX 1 - Gaussian mask instead of hard circular cutoff
        Ideal brick-wall mask -> sinc kernel in spatial domain -> Gibbs ringing.
        Gaussian mask rolls off smoothly -> zero ringing artefacts.

    FIX 2 - np.real() after IFFT, NOT np.abs()
        HPF output is a zero-mean *signed* signal (positive peaks / negative
        troughs around edges).  np.abs() folds negatives up -> fully-positive
        overlay -> illusion breaks.  np.real() keeps the sign intact.

    FIX 3 - Explicit DC removal from the high-pass component
        Subtracting the mean after IFFT guarantees mean == 0 exactly, preventing
        any brightness bias in the final blend.

    FIX 4 - Energy (std) balancing before blending
        std(high) is scaled to match std(low) before alpha/beta are applied, so
        the user weights are true artistic controls, not energy compensators.

    FEATURE 1 - Adaptive sigma
        lp_sigma and hp_sigma default to None.  When None, they are computed
        automatically from the image diagonal.

    FEATURE 2 - Grayscale preview mode
        When grayscale_preview=True (default), apply() returns a 2-D uint8 array.

    FEATURE 3 - Cached components
        After every call to _pipeline() the float64 intermediate arrays are stored
        as cache_low, cache_high, cache_hybrid.

    FEATURE 4 - RGB toggle
        get_rgb_preview(image) returns a colourised hybrid image.

    Parameters
    ----------
    image_high : np.ndarray
        The second image whose HIGH frequencies will be extracted.
    lp_sigma : float | None
        Sigma of the Gaussian low-pass mask.  None -> adaptive.
    hp_sigma : float | None
        Sigma of the Gaussian high-pass mask.  None -> adaptive.
    alpha : float
        Weight for the low-frequency component (default 1.0).
    beta : float
        Weight for the high-frequency component (default 1.0).
    grayscale_preview : bool
        If True (default), apply() returns 2-D grayscale.
    lp_ratio : float
        Fraction of image diagonal used as lp_sigma when lp_sigma is None.
    hp_ratio : float
        Fraction of image diagonal used as hp_sigma when hp_sigma is None.
    target_size : int
        Side length after center_crop_and_resize (default 512).
    """

    _LP_RATIO: float = 0.08
    _HP_RATIO: float = 0.04

    def __init__(
        self,
        image_high: np.ndarray,
        lp_sigma: float | None = None,
        hp_sigma: float | None = None,
        alpha: float = 1.0,
        beta: float = 1.8,
        grayscale_preview: bool = True,
        lp_ratio: float = 0.15,
        hp_ratio: float = 0.008,
        target_size: int = 512,
    ):
        self.image_high        = image_high
        self.lp_sigma          = lp_sigma
        self.hp_sigma          = hp_sigma
        self.alpha             = alpha
        self.beta              = beta
        self.grayscale_preview = grayscale_preview
        self.lp_ratio          = lp_ratio
        self.hp_ratio          = hp_ratio
        self.target_size       = target_size

        # FEATURE 3 - cache (populated after first _pipeline() call)
        self.cache_low:    np.ndarray | None = None
        self.cache_high:   np.ndarray | None = None
        self.cache_hybrid: np.ndarray | None = None
        self.effective_lp_sigma: float | None = None
        self.effective_hp_sigma: float | None = None

    # ------------------------------------------------------------------
    # FEATURE 1 - Adaptive sigma helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_diagonal(rows: int, cols: int) -> float:
        """Euclidean diagonal of the image in pixels."""
        return float(np.sqrt(rows ** 2 + cols ** 2))

    def _resolve_sigmas(self, rows: int, cols: int) -> tuple[float, float]:
        """Return (lp_sigma, hp_sigma) to use for this image size."""
        diag = self._compute_diagonal(rows, cols)
        lp = self.lp_sigma if self.lp_sigma is not None else diag * self.lp_ratio
        hp = self.hp_sigma if self.hp_sigma is not None else diag * self.hp_ratio
        return float(lp), float(hp)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_gray(image: np.ndarray) -> np.ndarray:
        """Convert BGR -> grayscale if needed; leave 2-D images untouched."""
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image.copy()

    @staticmethod
    def _gaussian_mask(rows: int, cols: int, sigma: float) -> np.ndarray:
        """
        2-D Gaussian low-pass mask centred at DC (post-fftshift).
        FIX 1: smooth roll-off -> no ringing artefacts.
        """
        crow, ccol = rows // 2, cols // 2
        v, u = np.ogrid[-crow:rows - crow, -ccol:cols - ccol]
        return np.exp(-(u * u + v * v) / (2.0 * sigma ** 2))

    @staticmethod
    def _fft_apply(channel: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        FFT -> mask -> IFFT -> np.real().
        FIX 2: real part only - preserves sign of HPF output.
        """
        f_shift  = np.fft.fftshift(np.fft.fft2(channel))
        filtered = np.fft.ifft2(np.fft.ifftshift(f_shift * mask))
        return np.real(filtered)

    def _extract_low(self, channel: np.ndarray, lp_sigma: float) -> np.ndarray:
        """Gaussian LPF: retains smooth global structure."""
        mask = self._gaussian_mask(*channel.shape, lp_sigma)
        return self._fft_apply(channel, mask)

    def _extract_high(self, channel: np.ndarray, hp_sigma: float) -> np.ndarray:
        """
        Gaussian HPF: 1 - LPF mask.
        FIX 3: subtract mean -> DC exactly 0.
        """
        lp_mask = self._gaussian_mask(*channel.shape, hp_sigma)
        hp_mask = 1.0 - lp_mask
        result  = self._fft_apply(channel, hp_mask)
        result -= result.mean()
        return result

    # ------------------------------------------------------------------
    # Core pipeline
    # ------------------------------------------------------------------

    def _pipeline(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Full hybrid pipeline.  Returns (low_comp, high_comp, hybrid) as float64.
        Results are also written to self.cache_* (FEATURE 3).
        """
        img_low  = center_crop_and_resize(self._to_gray(image),          self.target_size).astype(np.float64)
        img_high = center_crop_and_resize(self._to_gray(self.image_high), self.target_size).astype(np.float64)

        # FEATURE 1 - resolve adaptive sigmas
        lp_sigma, hp_sigma = self._resolve_sigmas(*img_low.shape)
        self.effective_lp_sigma = lp_sigma
        self.effective_hp_sigma = hp_sigma

        # Frequency separation
        low_comp  = self._extract_low(img_low,   lp_sigma)
        high_comp = self._extract_high(img_high, hp_sigma)

        # FIX 4: energy balancing
        std_low  = np.std(low_comp)
        std_high = np.std(high_comp)
        if std_high > 1e-8:
            high_comp = high_comp * (std_low / std_high)

        # Weighted blend
        hybrid = self.alpha * low_comp + self.beta * high_comp

        # FEATURE 3 - cache
        self.cache_low    = low_comp.copy()
        self.cache_high   = high_comp.copy()
        self.cache_hybrid = hybrid.copy()

        return low_comp, high_comp, hybrid

    # ------------------------------------------------------------------
    # Display helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _u8_positive(arr: np.ndarray) -> np.ndarray:
        """Shift a non-negative float array into [0, 255] uint8."""
        shifted = arr - arr.min()
        return np.clip(shifted, 0, 255).astype(np.uint8)

    @staticmethod
    def _u8_centred(arr: np.ndarray) -> np.ndarray:
        """Map a signed zero-mean array to uint8 with 0 -> 128."""
        return np.clip(arr + 128.0, 0, 255).astype(np.uint8)

    # ------------------------------------------------------------------
    # FEATURE 4 - RGB colourised output
    # ------------------------------------------------------------------

    def get_rgb_preview(self, image: np.ndarray) -> np.ndarray:
        """
        Return an RGB-colourised hybrid image (H x W x 3, uint8, BGR channel order).
        """
        low_comp, high_comp, hybrid = self._pipeline(image)

        src_low  = center_crop_and_resize(image,          self.target_size)
        src_high = center_crop_and_resize(self.image_high, self.target_size)

        if src_low.ndim == 2:
            src_low  = cv2.cvtColor(src_low,  cv2.COLOR_GRAY2BGR)
        if src_high.ndim == 2:
            src_high = cv2.cvtColor(src_high, cv2.COLOR_GRAY2BGR)

        e_low  = self.alpha  * (np.std(low_comp)  + 1e-8)
        e_high = self.beta   * (np.std(high_comp) + 1e-8)
        w_low  = e_low  / (e_low + e_high)
        w_high = e_high / (e_low + e_high)

        colour_blend = (
            w_low  * src_low.astype(np.float64) +
            w_high * src_high.astype(np.float64)
        ).astype(np.uint8)

        ycrcb = cv2.cvtColor(colour_blend, cv2.COLOR_BGR2YCrCb).astype(np.float64)
        hybrid_u8 = self._u8_positive(hybrid).astype(np.float64)
        ycrcb[:, :, 0] = hybrid_u8
        result_bgr = cv2.cvtColor(np.clip(ycrcb, 0, 255).astype(np.uint8),
                                  cv2.COLOR_YCrCb2BGR)
        return result_bgr

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def toggle_preview_mode(self, grayscale: bool) -> None:
        """Switch between grayscale and RGB output for apply()."""
        self.grayscale_preview = grayscale

    def apply(self, image: np.ndarray) -> np.ndarray:
        """
        Run the full hybrid pipeline and return the display image.
        
        Output mode is controlled by self.grayscale_preview:
          - True  (default) -> 2-D uint8 grayscale
          - False           -> H x W x 3 uint8 BGR colour
        """
        if self.grayscale_preview:
            _, _, hybrid = self._pipeline(image)
            return self._u8_positive(hybrid)
        else:
            return self.get_rgb_preview(image)

    def apply_extended(self, image: np.ndarray) -> dict[str, np.ndarray]:
        """
        Run the pipeline and return all intermediate display buffers.

        Keys: low_component, high_component, hybrid, hybrid_rgb,
              magnitude (alias), effective_sigmas (dict).
        """
        low_comp, high_comp, hybrid = self._pipeline(image)

        low_u8    = self._u8_positive(low_comp)
        high_u8   = self._u8_centred(high_comp)
        hybrid_u8 = self._u8_positive(hybrid)

        hybrid_rgb = self.get_rgb_preview(image)

        return {
            "low_component":   low_u8,
            "high_component":  high_u8,
            "hybrid":          hybrid_u8,
            "hybrid_rgb":      hybrid_rgb,
            "magnitude":       hybrid_u8,
            "effective_sigmas": {
                "lp": self.effective_lp_sigma,
                "hp": self.effective_hp_sigma,
            },
        }


# ==========================================
# 5) Global Enhancements
# ==========================================
class EqualizeHistogram(ImageOperation):
    def apply(self, image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 3:
            ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
            ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
            return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
        else:
            return cv2.equalizeHist(image)

class NormalizeImage(ImageOperation):
    def apply(self, image: np.ndarray) -> np.ndarray:
        return cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)


# ==========================================
# 6) Color Conversion
# ==========================================
class ManualGrayscale(ImageOperation):
    """
    Converts a color (BGR) image to grayscale using the ITU-R BT.601 luma formula:

        Gray = 0.299 * R + 0.587 * G + 0.114 * B
    """

    def apply(self, image: np.ndarray) -> np.ndarray:
        if image.ndim == 2:
            return image.copy()

        b = image[:, :, 0].astype(np.float64)
        g = image[:, :, 1].astype(np.float64)
        r = image[:, :, 2].astype(np.float64)

        gray = 0.299 * r + 0.587 * g + 0.114 * b

        return gray.astype(np.uint8)
