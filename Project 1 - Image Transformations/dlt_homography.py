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
