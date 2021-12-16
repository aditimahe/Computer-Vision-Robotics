import numpy as np
from numpy.linalg import inv, norm

def triangulate(Kl, Kr, Twl, Twr, pl, pr, Sl, Sr):
    """
    Triangulate 3D point position from camera projections.

    The function computes the 3D position of a point landmark from the 
    projection of the point into two camera images separated by a known
    baseline. All arrays should contain float64 values.

    Parameters:
    -----------
    Kl   - 3x3 np.array, left camera intrinsic calibration matrix.
    Kr   - 3x3 np.array, right camera intrinsic calibration matrix.
    Twl  - 4x4 np.array, homogeneous pose, left camera in world frame.
    Twr  - 4x4 np.array, homogeneous pose, right camera in world frame.
    pl   - 2x1 np.array, point in left camera image.
    pr   - 2x1 np.array, point in right camera image.
    Sl   - 2x2 np.array, left image point covariance matrix.
    Sr   - 2x2 np.array, right image point covariance matrix.

    Returns:
    --------
    Pl  - 3x1 np.array, closest point on ray from left camera  (in world frame).
    Pr  - 3x1 np.array, closest point on ray from right camera (in world frame).
    P   - 3x1 np.array, estimated 3D landmark position in the world frame.
    S   - 3x3 np.array, covariance matrix for estimated 3D point.
    """
    #--- FILL ME IN ---

    # Compute baseline (right camera translation minus left camera translation).

    # Unit vectors projecting from optical center to image plane points.
    # Use variables rayl and rayr for the rays.
 
    # Projected segment lengths.
    # Use variables ml and mr for the segment lengths.
    
    # Segment endpoints.
    # User variables Pl and Pr for the segment endpoints.

    # Now fill in with appropriate ray Jacobians. These are 
    # 3x4 matrices, but two columns are zeros (because the right
    # ray direction is not affected by the left image point and 
    # vice versa).
    drayl = np.zeros((3, 4))  # Jacobian left ray w.r.t. image points.
    drayr = np.zeros((3, 4))  # Jacobian right ray w.r.t. image points.

    # Add code here...

    # Computing the baseline (right camera translation minus left camera
    # translation) reshaped to a 3 by 1 vector
    b = (Twr[:3, 3] - Twl[:3, 3]).reshape(3, 1)

    # Compute the unit vectors projecting from optical center to image plane points
    rL = Twl[:3, :3] @ inv(Kl) @ np.vstack((pl, 1))
    rR = Twr[:3, :3] @ inv(Kr) @ np.vstack((pr, 1))
    rayl = rL / norm(rL)
    rayr = rR / norm(rR)

    # Calculate projected segment scalar lengths
    ml = (np.dot(b.T, rayl) - (np.dot(b.T, rayr) * np.dot(rayl.T, rayr))) / (1 - (np.dot(rayl.T, rayr))**2)
    mr = (np.dot(rayl.T, rayr) * ml) - np.dot(b.T, rayr)

    # Calculating the coordinates of the line segment endpoints
    Pl = Twl[:3, 3].reshape((3, 1)) + (rayl * ml)
    Pr = Twr[:3, 3].reshape(3, 1) + (rayr * mr)

    # Obtain the Jacobian of rL and rR
    drL = Twl[:3, :3] @ inv(Kl)
    drR = Twr[:3, :3] @ inv(Kr)
    drL = np.hstack((drL[:, 0:2], np.zeros((3, 2))))
    drR = np.hstack((np.zeros((3, 2)), drR[:, 0:2]))

    # Obtain the Jacobian of the norm of rL and rR
    drLNorm = (1 / norm(rL)) * (np.transpose(rL) @ Twl[:3, :3] @ inv(Kl))
    drRNorm = (1 / norm(rR)) * (np.transpose(rR) @ Twr[:3, :3] @ inv(Kr))
    drLNorm = np.hstack((drLNorm[:, 0:2], np.zeros((1, 2))))
    drRNorm = np.hstack((np.zeros((1, 2)), drRNorm[:, 0:2]))

    # Use the equation to obtain the ray Jacobians
    drayl = (drL * norm(rL) - rL * drLNorm) / (norm(rL)**2)
    drayr = (drR * norm(rR) - rR * drRNorm) / (norm(rR)**2)

    #------------------

    # Compute dml and dmr (partials wrt segment lengths).
    u = np.dot(b.T, rayl) - np.dot(b.T, rayr)*np.dot(rayl.T, rayr)
    v = 1 - np.dot(rayl.T, rayr)**2

    du = (b.T@drayl).reshape(1, 4) - \
         (b.T@drayr).reshape(1, 4)*np.dot(rayl.T, rayr) - \
         np.dot(b.T, rayr)*((rayr.T@drayl) + (rayl.T@drayr)).reshape(1, 4)
 
    dv = -2*np.dot(rayl.T, rayr)*((rayr.T@drayl).reshape(1, 4) + \
        (rayl.T@drayr).reshape(1, 4))

    m = np.dot(b.T, rayr) - np.dot(b.T, rayl)@np.dot(rayl.T, rayr)
    n = np.dot(rayl.T, rayr)**2 - 1

    dm = (b.T@drayr).reshape(1, 4) - \
         (b.T@drayl).reshape(1, 4)*np.dot(rayl.T, rayr) - \
         np.dot(b.T, rayl)@((rayr.T@drayl) + (rayl.T@drayr)).reshape(1, 4)
    dn = -dv

    dml = (du*v - u*dv)/v**2
    dmr = (dm*n - m*dn)/n**2

    # Finally, compute Jacobian for P w.r.t. image points.
    JP = (ml*drayl + rayl*dml + mr*drayr + rayr*dmr)/2

    #--- FILL ME IN ---

    # 3D point.
    # Obtaining the estimated 3D landmark position in the world frame
    P = (Pl + Pr) / 2

    # 3x3 landmark point covariance matrix (need to form
    # the 4x4 image plane covariance matrix first).

    # Obtaining the 3x3 landmark point covariance matrix using info from above
    mTop = np.hstack((Sl, np.zeros((2, 2))))
    mBot = np.hstack((np.zeros((2, 2)), Sr))
    S = JP @ np.vstack((mTop, mBot)) @ np.transpose(JP)


    #------------------



    # Check for correct outputs...
    correct = isinstance(Pl, np.ndarray) and Pl.shape == (3, 1) and \
              isinstance(Pr, np.ndarray) and Pr.shape == (3, 1) and \
              isinstance(P,  np.ndarray) and P.shape  == (3, 1) and \
              isinstance(S,  np.ndarray) and S.shape  == (3, 3)

    if not correct:
        raise TypeError("Wrong type or size returned!")

    return Pl, Pr, P, S