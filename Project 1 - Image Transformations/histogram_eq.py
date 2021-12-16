import numpy as np

def histogram_eq(I):
    """
    Histogram equalization for greyscale image.

    Perform histogram equalization on the 8-bit greyscale intensity image I
    to produce a contrast-enhanced image J. Full details of the algorithm are
    provided in the Szeliski text.

    Parameters:
    -----------
    I  - Single-band (greyscale) intensity image, 8-bit np.array (i.e., uint8).

    Returns:
    --------
    J  - Contrast-enhanced greyscale intensity image, 8-bit np.array (i.e., uint8).
    """
    #--- FILL ME IN ---

    # Verify I is grayscale.
    if I.dtype != np.uint8:
        raise ValueError('Incorrect image format!')

    # Compute the histogram of the image with range of 0 to L - 1, where L = 256
    imgHistogram, bins = np.histogram(I.flatten(), 256, [0, 256])

    # Normalize the histogram
    imgHistogram = imgHistogram / (np.sum(imgHistogram))

    # Compute the cumulative distribution function for the image histogram using the cumsum function
    cumSum = imgHistogram.cumsum()

    # Determine the mapping function to transform the pixels
    transformMap = np.round(255 * cumSum).astype('uint8')

    # Flattening the image to a list for mapping
    li = list(I.flatten())

    # Transforming the pixel values
    J = [transformMap[i] for i in li]

    # Reshape the the transformed pixels into the appropriate size
    J = np.reshape(np.asarray(J), I.shape)

    #------------------

    return J
