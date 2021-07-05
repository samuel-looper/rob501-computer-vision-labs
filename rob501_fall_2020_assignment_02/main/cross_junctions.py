import numpy as np
from numpy.linalg import inv, lstsq
from scipy.linalg import null_space
from scipy.ndimage.filters import *
from matplotlib.path import Path
import matplotlib.pyplot as plt

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

    # Calculate bounds on World representation points for Direct Linear Transform
    x_dist = np.unique(np.sort(Wpts[0]))[1] - np.unique(np.sort(Wpts[0]))[0]
    x_min = np.unique(np.sort(Wpts[0]))[0] - x_dist*1.25
    x_max = np.unique(np.sort(Wpts[0]))[-1] + x_dist*1.25
    y_dist = np.unique(np.sort(Wpts[1]))[1] - np.unique(np.sort(Wpts[1]))[0]
    y_min = np.unique(np.sort(Wpts[1]))[0] - y_dist*1.25
    y_max = np.unique(np.sort(Wpts[1]))[-1] + y_dist*1.25

    I1pts = np.asarray([[x_min, x_max, x_max, x_min],
                        [y_min, y_min, y_max, y_max]])
    I2pts = bpoly

    # Calculate Direct Linear Transform (from A1)
    A = np.zeros((8, 9))
    for i in range(len(I1pts[0])):
        x = I1pts[0][i]
        y = I1pts[1][i]
        u = I2pts[0][i]
        v = I2pts[1][i]
        A[2 * i] = [-x, -y, -1, 0, 0, 0, u * x, u * y, u]
        A[2 * i + 1] = [0, 0, 0, -x, -y, -1, v * x, v * y, v]
    if (null_space(A).shape) != (9, 1):
        return "ERROR"
    H = np.reshape(null_space(A), (3, 3))

    # Compute estimate of cross_junctions using Direct Linear Transform
    Wpts[2, :] = np.ones((1, Wpts.shape[1]))
    est_pts = np.dot(H, Wpts)
    est_pts_norm = est_pts[:2, :]/est_pts[2, :]

    # Refinement and final matrix construction
    for i in range(est_pts_norm.shape[1]):
        x_val = np.round(est_pts_norm[0, i]).astype(int) # integer value for saddle point algorithm bounds
        y_val = np.round(est_pts_norm[1, i]).astype(int)
        I_crop = gaussian_filter(I[y_val-20:y_val+20, x_val-20:x_val+20], 1) # Image for saddle point algorithm

        # Saddle point algorithm (see saddle_point for more info)
        indx = (np.indices(I_crop.shape))
        ys = np.ravel(indx[0, :, :])
        xs = np.ravel(indx[1, :, :])
        flatten_I = np.ravel(I_crop)
        aug_I = np.zeros((I_crop.size, 6))
        for j in range(I_crop.size):
            x = xs[j]
            y = ys[j]
            aug_I[j, :] = np.asarray([x ** 2, y * x, y ** 2, x, y, 1])

        params, residuals, rank, s = np.linalg.lstsq(aug_I, flatten_I, rcond=None)
        param_mat = np.asarray([[2 * params[0], params[1]], [params[1], 2 * params[2]]])
        optim = -np.dot(np.linalg.inv(param_mat), np.asarray([[params[3], params[4]]]).T)

        # Adjust estimated values based on saddle point algorithm output
        x_adjust = optim[0, 0]-20 + x_val
        y_adjust = optim[1, 0]-20 + y_val
        est_pts_norm[:, i] = [x_adjust, y_adjust]

    Ipts = est_pts_norm
    #------------------

    correct = isinstance(Ipts, np.ndarray) and \
        Ipts.dtype == np.float64 and Ipts.shape[0] == 2

    if not correct:
        raise TypeError("Wrong type or size returned!")

    return Ipts