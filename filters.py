import numpy as np

def compute_histogram(image):
    """
    Calculates the frequency of each pixel intensity.
    Input: Grayscale image (2D numpy array)
    Output: List/Array of 256 values
    """
    # 1. Create an array of 256 zeros to store counts
    hist = np.zeros(256, dtype=int)
    
    # 2. Flatten the 2D image into a 1D list of pixels
    pixels = image.flatten()
    
    # 3. Loop through pixels and increment the corresponding index
    for pixel in pixels:
        hist[pixel] += 1
        
    return hist

def compute_cdf(hist):
    """
    Calculates the Cumulative Distribution Function.
    Input: Histogram (list of 256 values)
    Output: Normalized CDF (0.0 to 1.0)
    """
    # Cumulative sum: [a, a+b, a+b+c, ...]
    cdf = hist.cumsum()
    
    # Normalize to keep it between 0 and 1 (as requested by TA)
    cdf_normalized = cdf / cdf.max()
    
    return cdf_normalized
