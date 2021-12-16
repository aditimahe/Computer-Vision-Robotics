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