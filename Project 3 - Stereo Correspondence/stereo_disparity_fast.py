import numpy as np
from scipy.ndimage.filters import *

def stereo_disparity_fast(Il, Ir, bbox, maxd):
    """
    Fast stereo correspondence algorithm.

    This function computes a stereo disparity image from left stereo 
    image Il and right stereo image Ir. Only disparity values within
    the bounding box region (inclusive) are evaluated.

    Parameters:
    -----------
    Il    - Left stereo image, m x n pixel np.array, greyscale.
    Ir    - Right stereo image, m x n pixel np.array, greyscale.
    bbox  - 2x2 np.array, bounding box, relative to left image, from top left
            corner to bottom right corner (inclusive). (x, y) in columns.
    maxd  - Integer, maximum disparity value; disparities must be within zero
            to maxd inclusive (i.e., don't search beyond maxd).

    Returns:
    --------
    Id  - Disparity image (map) as np.array, same size as Il.
    """
    # Hints:
    #
    #  - Loop over each image row, computing the local similarity measure, then
    #    aggregate. At the border, you may replicate edge pixels, or just avoid
    #    using values outside of the image.
    #
    #  - You may hard-code any parameters you require in this function.
    #
    #  - Use whatever window size you think might be suitable.
    #
    #  - Optimize for runtime AND for clarity.

    #--- FILL ME IN ---

    # Code goes here...
    # Initiate disparity image map (same size as Il)
    Id = np.zeros_like(Il)

    # Set window size, optimized after multiple iterations
    windowSize = 7

    # Pad images (replicating edge pixels) using numpy's pad function
    IlPad = np.pad(Il, round(windowSize/2), 'edge')
    IrPad = np.pad(Ir, round(windowSize/2), 'edge')

    # Determine height and width of the padded image to be used later
    heightPad, widthPad = np.shape(IlPad)

    # Define the ranges we need to search over from the bbox
    top = bbox[1, 0]
    bottom = bbox[1, 1]
    left = bbox[0, 0]
    right = bbox[0, 1]

    # Loop through each row of the image
    for i in range(top, bottom + 1):
        # print('i: ', i)

        # Loop through each column (each pixel) of the image
        for j in range(left, right + 1):
            # Define the patch in question from the left image
            leftImagePatch = IlPad[i:i + windowSize, j:j + windowSize]

            # Initiate best SAD measure as infinity
            sadMeasuresBest = np.inf

            # Initiate best disparity as None
            bestDisp = None

            # Loop through all possible disparity values (negative to positive)
            for d in range(-maxd, maxd):
                # Checking that we are within bounds (i.e. < widthPad and >= 0)
                if((j + d + windowSize) < widthPad and (j + d) >= 0):
                    # Define the patch in question from the right image
                    rightImagePatch = IrPad[i:(i + windowSize), (j + d):(j + d + windowSize)]

                    # Obtain SAD measure for the patches using the equation
                    sadMeasures = np.sum(np.abs(leftImagePatch - rightImagePatch))

                    # If new SAD measure is < than the best one, save the SAD measure and disparity as best
                    if sadMeasures < sadMeasuresBest:
                        sadMeasuresBest = sadMeasures
                        bestDisp = abs(d)

            # Assign the best disparity to the disparity image map to be returned
            Id[i, j] = bestDisp

    #------------------

    correct = isinstance(Id, np.ndarray) and Id.shape == Il.shape

    if not correct:
        raise TypeError("Wrong type or size returned!")

    return Id