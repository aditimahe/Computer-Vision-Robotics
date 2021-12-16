import numpy as np
from scipy.ndimage.filters import *

def stereo_disparity_best(Il, Ir, bbox, maxd):
    """
    Best stereo correspondence algorithm.

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

    # I implemented a sharpen filter on both the left and right input images to highlight edges and enhance features.
    # This type of preprocessing allows the enhanced information to be compared between the images. Image sharpening
    # is used quite often in computer vision and has proven to improve the results.
    # The disparity image map from the fast algorithm is quite noisy when compared to the ground truth. One can see
    # that there are a lot of small mismatched/white patches. The Literature Survey on Stereo Vision Disparity Map
    # Algorithms found here (https://www.hindawi.com/journals/js/2016/8742920/) suggested using a median filter for
    # disparity map refinement. According to the literature survey, the median filter is able to
    # remove small and isolated mismatches in disparity (i.e. noise that was visible from the fast algorithm).
    # To further improve the results, I experimented and found that using a percentile filter on top of the median
    # filter works well.


    # Initiate disparity image map (same size as Il)
    Id = np.zeros_like(Il)

    # Set window size, optimized after multiple iterations
    windowSize = 7

    # Define the sharpen kernel and use numpy's convolve function to sharpen both images
    sharpenKernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    IlSharp = convolve(Il, sharpenKernel, mode = 'constant')
    IrSharp = convolve(Ir, sharpenKernel, mode = 'constant')

    # Pad images (replicating edge pixels) using numpy's pad function
    IlPad = np.pad(IlSharp, round(windowSize/2), 'edge')
    IrPad = np.pad(IrSharp, round(windowSize/2), 'edge')
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

    # Apply the median filter and percentile filter for disparity map refinement
    Id[top:(bottom + 1), left:(right + 1)] = median_filter(Id[top:(bottom + 1), left:(right + 1)], size = 13)
    Id[top:(bottom + 1), left:(right + 1)] = percentile_filter(Id[top:(bottom + 1), left:(right + 1)], percentile = 40, size = 15)
    #------------------

    correct = isinstance(Id, np.ndarray) and Id.shape == Il.shape

    if not correct:
        raise TypeError("Wrong type or size returned!")

    return Id