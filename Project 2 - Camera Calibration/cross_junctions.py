import numpy as np
from numpy.linalg import inv, lstsq
from scipy.linalg import null_space
from scipy.ndimage.filters import *
from matplotlib.path import Path

# You may add support functions here, if desired.

def cross_junctions(I, bpoly, Wpts):
    """
    Find cross-junctions in image with subpixel accuracy.

    The function locates a series of cross-junction points on a planar 
    calibration target, where the target is bounded in the image by the 
    specified quadrilateral. The number of cross-junctions identified 
    should be equal to the number of world points.

    Note also that the world and image points must be in *correspondence*,
    that is, the first world point should map to the first image point, etc.

    Parameters:
    -----------
    I      - Single-band (greyscale) image as np.array (e.g., uint8, float).
    bpoly  - 2x4 np.array, bounding polygon (clockwise from upper left).
    Wpts   - 3xn np.array of world points (in 3D, on calibration target).

    Returns:
    --------
    Ipts  - 2xn np.array of cross-junctions (x, y), relative to the upper
            left corner of the target. The array must contain float64 values.
    """
    #--- FILL ME IN ---

    # Checkboard square size in meters
    checkerSquare = 63.5 / 1000.0

    # Determine the distance from nearest cross junction to the border in x and y directions
    # It is the length of one checkerboard square plus the fraction of the square for each direction
    # Guess and checked the results to obtain the most appropriate ratio
    distXJunctionToBorder = checkerSquare + ((1/3) * checkerSquare)
    distYJunctionToBorder = checkerSquare + ((1/5) * checkerSquare)

    # Obtain the coordinates (from the world points) of the closest cross-junction points to the borders
    # and use those to determine the coordinates of the border
    upperLeft = Wpts[:, 0:1] + np.array([[-distXJunctionToBorder], [-distYJunctionToBorder], [0]])
    upperRight = Wpts[:, 7:8] + np.array([[distXJunctionToBorder], [-distYJunctionToBorder], [0]])
    lowerRight = Wpts[:, 47:48] + np.array([[distXJunctionToBorder], [distYJunctionToBorder], [0]])
    lowerLeft = Wpts[:, 40:41] + np.array([[-distXJunctionToBorder], [distYJunctionToBorder], [0]])

    # Combine the coordinates in appropriate input for homography
    I1Wpts = np.hstack((upperLeft[:2], upperRight[:2], lowerRight[:2], lowerLeft[:2]))

    # Use the dlt_homography function from the last assignment to perform homography from
    # the world points to the bounding polygon
    H, A = dlt_homography(I1Wpts, bpoly)

    # Calculate the homography cross junctions using the H matrix and normalizing
    toHomography = Wpts
    toHomography[2, :] = 1
    crossJunctionsH = H @ toHomography
    n = (H @ toHomography)[2, :]
    crossJunctionsH = crossJunctionsH / n
    crossJunctionsH = crossJunctionsH[:2]

    # Loop through each cross junction point
    # Use saddle point function by passing image patch of 20 to get appropriate locations
    # Add to proper sized numpy array for returning 2xn np.array of cross-junctions (x, y)
    Ipts = np.array([[], []])
    for i in range(crossJunctionsH.shape[1]):
        x = int(crossJunctionsH[0][i])
        y = int(crossJunctionsH[1][i])
        saddlePoint = saddle_point(I[y - 10:y + 10, x - 10:x + 10])
        saddlePointX = saddlePoint[0, 0]
        saddlePointY = saddlePoint[1, 0]
        toAdd = np.array([[x - 10 + saddlePointX], [y - 10 + saddlePointY]])
        Ipts = np.hstack((Ipts, toAdd))

    #------------------

    correct = isinstance(Ipts, np.ndarray) and \
        Ipts.dtype == np.float64 and Ipts.shape[0] == 2

    if not correct:
        raise TypeError("Wrong type or size returned!")

    return Ipts


import numpy as np
from numpy.linalg import inv, norm
from scipy.linalg import null_space

def dlt_homography(I1pts, I2pts):
    """
    Find perspective Homography between two images.

    Given 4 points from 2 separate images, compute the perspective homography
    (warp) between these points using the DLT algorithm.

    Parameters:
    -----------
    I1pts  - 2x4 np.array of points from Image 1 (each column is x, y).
    I2pts  - 2x4 np.array of points from Image 2 (in 1-to-1 correspondence).

    Returns:
    --------
    H  - 3x3 np.array of perspective homography (matrix map) between image coordinates.
    A  - 8x9 np.array of DLT matrix used to determine homography.
    """
    #--- FILL ME IN ---

    # Initialize A matrix (8x9 np.array)
    A = np.zeros((8,9))

    # To fill in the A matrix, run the for loop for each of the points (columns)
    for i in range(I1pts.shape[1]):
        # Determine the 2 corresponding points (representative of x = [x y 1]^T and x' = [u v 1]^T from paper)
        # to be used for the Direct Linear Transform (DLT)
        x = I1pts[0, i]
        y = I1pts[1, i]
        u = I2pts[0, i]
        v = I2pts[1, i]

        # Populate the A matrix given on pg. 11 Section 2.1 of the M.A.Sc. thesis written by ElanDubrofsky of UBC
        A[i*2, :] = [-x, -y, -1, 0, 0, 0, u*x, u*y, u]
        A[i*2 + 1, :] = [0, 0, 0, -x, -y, -1, v*x, v*y, v]

    # Solve for the solution space (nullspace of A is the solution space for h)
    h = null_space(A)

    # Reshape the solution to a 3x3 np.array of perspective homography (matrix map), H
    H = np.reshape(h, (3,3))

    # Normalize the matrix map by scaling each entry so that the lower right entry [2, 2] is 1
    H = H / H[2, 2]

    #------------------

    return H, A

import numpy as np
from numpy.linalg import inv, lstsq

def saddle_point(I):
    """
    Locate saddle point in an image patch.

    The function identifies the subpixel centre of a cross-junction in the
    image patch I, by fitting a hyperbolic paraboloid to the patch, and then
    finding the critical point of that paraboloid.

    Note that the location of 'pt' is relative to (-0.5, -0.5) at the upper
    left corner of the patch, i.e., the pixels are treated as covering an
    area of one unit square.

    Parameters:
    -----------
    I  - Single-band (greyscale) image patch as np.array (e.g., uint8, float).

    Returns:
    --------
    pt  - 2x1 np.array (float64), subpixel location of saddle point in I (x, y).
    """
    #--- FILL ME IN ---
    # Saddle points are solved for using the paper outlined in the assignment pdf

    # Obtain shape of image patch (m rows, n columns)
    m, n = np.shape(I)

    # Initialize the A matrix and b vector for lstsq function
    A = np.zeros((m*n, 6))
    b = np.zeros((m*n, 1))

    # Initialize count variable for filling in A matrix and b vector appropriately
    c = 0

    # Loop through each pixel in the image patch and fill in A matrix and b vector
    for x in range(m):
        for y in range(n):
            A[c, :] = [x*x, x*y, y*y, x, y, 1]
            b[c] = I[y, x]
            c = c + 1

    # Solve for the coefficients using numpy's lstsq function
    result = lstsq(A, b, rcond = None)[0]
    alpha, beta, gamma, delta, epsilon, zeta = list(result)
    alpha, beta, gamma, delta, epsilon, zeta = alpha[0], beta[0], gamma[0], delta[0], epsilon[0], zeta[0]

    # Solve for the point using the solved coefficients (determined as described in the paper)
    # Reshape appropriately
    M = np.array([[2*alpha, beta], [beta, 2*gamma]])
    v = np.array([delta, epsilon])
    pt = np.dot((-1*inv(M)), v)
    pt = pt.reshape((2, 1))

    #------------------

    correct = isinstance(pt, np.ndarray) and \
              pt.dtype == np.float64 and pt.shape == (2, 1)

    if not correct:
        raise TypeError("Wrong type or size returned!")

    return pt