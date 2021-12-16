# Billboard hack script file.
import numpy as np
from matplotlib.path import Path
from imageio import imread, imwrite

from dlt_homography import dlt_homography
from bilinear_interp import bilinear_interp
from histogram_eq import histogram_eq

def billboard_hack():
    """
    Hack and replace the billboard!

    Parameters:
    ----------- 

    Returns:
    --------
    Ihack  - Hacked RGB intensity image, 8-bit np.array (i.e., uint8).
    """
    # Bounding box in Y & D Square image.
    bbox = np.array([[404, 490, 404, 490], [38,  38, 354, 354]])

    # Point correspondences.
    Iyd_pts = np.array([[416, 485, 488, 410], [40,  61, 353, 349]])
    Ist_pts = np.array([[2, 218, 218, 2], [2, 2, 409, 409]])

    Iyd = imread('../billboard/yonge_dundas_square.jpg')
    Ist = imread('../billboard/uoft_soldiers_tower_dark.png')

    Ihack = np.asarray(Iyd)
    Ist = np.asarray(Ist)

    #--- FILL ME IN ---

    # Let's do the histogram equalization first.
    # Histogram equalization on the uoft soldiers tower image
    stImageHistEq = histogram_eq(Ist)

    # Compute the perspective homography we need...
    # for points from yonge_dundas_square.jpg and uoft_soldiers_tower_dark.png
    H, A = dlt_homography(Iyd_pts, Ist_pts)

    # Main 'for' loop to do the warp and insertion -
    # this could be vectorized to be faster if needed!

    # You may wish to make use of the contains_points() method
    # available in the matplotlib.path.Path class!

    # Convert the billboard points into a polygon using matplotlib.path.Path
    billBoard = Path(Iyd_pts.T)

    # Obtain the min and max points for x and y to loop through from the bbox
    xMin = np.min(bbox[0])
    xMax = np.max(bbox[0])
    yMin = np.min(bbox[1])
    yMax = np.max(bbox[1])

    # Loop through each of the pixels in the bounding box
    for x in range(xMin, xMax + 1):
        for y in range(yMin, yMax + 1):
            # Check to see if the x and y point is within the billboard
            # If yes, we proceed with warp and insertion
            if billBoard.contains_point(np.array([x, y])) == True:
                # Obtain the point in homogeneous form and perform multiplication by the H matrix
                homogeneousPoint = np.array([x, y, 1.0])
                projTransHomogeneous = np.dot(H, homogeneousPoint)

                # Convert back from homogeneous form to inhomogeneous form by dividing by the last element
                projTrans = projTransHomogeneous / projTransHomogeneous[2]

                # Reshape appropriately
                a = projTrans[:2].reshape(2, 1)

                # Perform bilinear interpolation to obtain appropriate pixel intensity
                pixelIntensity = bilinear_interp(stImageHistEq, a)

                # Assign the pixel intensity in a 3 by 1 array for RGB to Ihack
                Ihack[y, x] = np.array([pixelIntensity, pixelIntensity, pixelIntensity])

    #------------------

    # plt.imshow(Ihack)
    # plt.show()
    # imwrite(Ihack, 'billboard_hacked.png');

    return Ihack