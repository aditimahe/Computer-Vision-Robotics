import numpy as np
from numpy.linalg import inv

def bilinear_interp(I, pt):
    """
    Performs bilinear interpolation for a given image point.

    Given the (x, y) location of a point in an input image, use the surrounding
    4 pixels to conmpute the bilinearly-interpolated output pixel intensity.

    Note that images are (usually) integer-valued functions (in 2D), therefore
    the intensity value you return must be an integer (use round()).

    This function is for a *single* image band only - for RGB images, you will 
    need to call the function once for each colour channel.

    Parameters:
    -----------
    I   - Single-band (greyscale) intensity image, 8-bit np.array (i.e., uint8).
    pt  - 2x1 np.array of point in input image (x, y), with subpixel precision.

    Returns:
    --------
    b  - Interpolated brightness or intensity value (whole number >= 0).
    """
    #--- FILL ME IN ---

    if pt.shape != (2, 1):
        raise ValueError('Point size is incorrect.')

    # Note: y is the rows of the and x is the columns

    # Obtain the x and y of the pt parameter
    x = pt[0, 0]
    y = pt[1, 0]

    # Obtain the coordinates of the 4 surrounding pixels
    # x0 and y0 are obtained by using floor to get the largest integer such that the integer is less than or equal to x or y
    # x1 and y1 are obtained by adding 1 to the corresponding x0 and y0 to to get the next pixels
    x0 = np.floor(x).astype(int)
    y0 = np.floor(y).astype(int)
    x1 = x0 + 1
    y1 = y0 + 1

    # Clip the coordinates such that they don't go beyond the pixels in the image
    x0 = np.clip(x0, 0, I.shape[1]-1)
    y0 = np.clip(y0, 0, I.shape[0]-1)
    x1 = np.clip(x1, 0, I.shape[1]-1)
    y1 = np.clip(y1, 0, I.shape[0]-1)

    # Obtain the pixel intensities of the 4 surrounding pixels
    pi_11 = I[y0, x0]
    pi_21 = I[y0, x1]
    pi_12 = I[y1, x0]
    pi_22 = I[y1, x1]

    # Use the bilinear interpolation formula to calculate b
    b = (x1 - x) * (y1 - y) * pi_11 + (x - x0) * (y1 - y) * pi_21 + (x1 - x) * (y - y0) * pi_12 + (x - x0) * (y - y0) * pi_22
    b = b / ((x1 - x0) * (y1 - y0))

    # Round and cast b to be an int
    b = int(np.round(b))

    #------------------

    return b
