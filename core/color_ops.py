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

def normalize_image(image):
    """
    Stretches the image intensity to the full 0-255 range.
    """
    # Convert to float to avoid overflow during calculation
    img_float = image.astype(float)
    
    i_min = np.min(img_float)
    i_max = np.max(img_float)
    
    # Avoid division by zero if the image is a solid color
    if i_max == i_min:
        return image
        
    normalized = ((img_float - i_min) / (i_max - i_min)) * 255
    
    return normalized.astype(np.uint8)


def equalize_image(image):
    """
    Applies histogram equalization to improve contrast.
    """
    # 1. Get the histogram and CDF (using your existing functions)
    hist = compute_histogram(image)
    # We need the non-normalized CDF for the standard formula
    cdf = hist.cumsum() 
    
    # 2. Mask the zero values to find the true cdf_min
    cdf_m = np.ma.masked_equal(cdf, 0)
    
    # 3. Apply the equalization formula
    # This creates a look-up table (LUT)
    num = (cdf_m - cdf_m.min()) * 255
    den = (cdf_m.max() - cdf_m.min())
    cdf_m = num / den
    
    # 4. Fill masked values back with 0 and convert to uint8
    final_cdf = np.ma.filled(cdf_m, 0).astype('uint8')
    
    # 5. Map the original pixels to the new equalized values
    # This is a very fast way to apply the mapping in NumPy
    equalized_img = final_cdf[image]
    
    return equalized_img
