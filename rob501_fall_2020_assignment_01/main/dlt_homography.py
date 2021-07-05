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
    H = np.zeros((3, 3))
    A = np.zeros((8,9))
    for i in range(len(I1pts[0])):
        x = I1pts[0][i]
        y = I1pts[1][i]
        u = I2pts[0][i]
        v = I2pts[1][i]
        A[2*i] = [-x, -y, -1, 0, 0 , 0, u*x, u*y, u]
        A[2*i+1] = [0, 0, 0, -x, -y, -1, v*x, v*y, v]
    if (null_space(A).shape) != (9,1):
        return "ERROR"
    H = np.reshape(null_space(A), (3,3))
    #------------------

    return H, A