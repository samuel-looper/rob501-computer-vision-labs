import numpy as np
from numpy.linalg import inv, norm
from find_jacobian import find_jacobian
from dcm_from_rpy import dcm_from_rpy
from rpy_from_dcm import rpy_from_dcm

#----- Functions Go Below -----

def epose_from_hpose(T):
    """Covert 4x4 homogeneous pose matrix to x, y, z, roll, pitch, yaw."""
    E = np.zeros((6, 1))
    E[0:3] = np.reshape(T[0:3, 3], (3, 1))
    E[3:6] = rpy_from_dcm(T[0:3, 0:3])
  
    return E

def hpose_from_epose(E):
    """Covert x, y, z, roll, pitch, yaw to 4x4 homogeneous pose matrix."""
    T = np.zeros((4, 4))
    T[0:3, 0:3] = dcm_from_rpy(E[3:6])
    T[0:3, 3] = np.reshape(E[0:3], (3,))
    T[3, 3] = 1
  
    return T

def pose_estimate_nls(K, Twc_guess, Ipts, Wpts):
    """
    Estimate camera pose from 2D-3D correspondences via NLS.

    The function performs a nonlinear least squares optimization procedure 
    to determine the best estimate of the camera pose in the calibration
    target frame, given 2D-3D point correspondences.

    Parameters:
    -----------
    K          - 3x3 camera intrinsic calibration matrix.
    Twc_guess  - 4x4 homogenous pose matrix, initial guess for camera pose.
    Ipts       - 2xn array of cross-junction points (with subpixel accuracy).
    Wpts       - 3xn array of world points (one-to-one correspondence with Ipts).

    Returns:
    --------
    Twc  - 4x4 np.array (float64), pose matrix, camera pose in target frame.
    """
    maxIters = 250                          # Set maximum iterations.

    tp = Ipts.shape[1]                      # Num points.
    ps = np.reshape(Ipts, (2*tp, 1), 'F')   # Stacked vector of observations.

    dY = np.zeros((2*tp, 1))                # Residuals.
    J  = np.zeros((2*tp, 6))                # Jacobian.

    #--- FILL ME IN ---

    # Some hints on structure are included below...

    # 1. Convert initial guess to parameter vector (6 x 1).
    params = epose_from_hpose(Twc_guess)

    iter = 1

    # 2. Main loop - continue until convergence or maxIters.
    while True:
        params_prev = params # Save previous best pose estimate.
        Twc_guess = hpose_from_epose(params_prev)

        # Build Error Vector and Jacobian Matrix.
        e_mat = np.zeros((2*tp, 1)) # (2n, 1) Error Vector
        big_J = np.zeros((2*tp, 6)) # (2n, 6) Jacobian Matrix
        for i in np.arange(tp):
            pt = np.asarray([Wpts[:, i]]).T
            p_world = np.vstack([pt, [1]])
            x_pred = np.dot(K, np.dot(inv(Twc_guess), p_world)[0:3]) # Calculate predicted image point projection
            e_mat[2*i, 0] = Ipts[0, i] - x_pred[0]/x_pred[2]
            e_mat[2*i+1, 0] = Ipts[1, i] - x_pred[1]/x_pred[2] # Add prediction error to error vector
            J = find_jacobian(K, Twc_guess, pt) # Calculate Jacobian for given image point
            big_J[2*i, :] = J[0, :]
            big_J[2*i + 1, :] = J[1, :] # Add jacobian to Jacobian matrix

        # Update parameters based on Error and Jacobian Matrix according to NLS algorithm
        # Note: + sign for camera to world params
        params = params_prev + np.dot(inv(np.dot(big_J.T, big_J)), np.dot(big_J.T, e_mat))

        # Check convergence
        diff = norm(params - params_prev)

        if norm(diff) < 1e-12:
            print("Covergence required %d iters." % iter)
            break
        elif iter == maxIters:
            print("Failed to converge after %d iters." % iter)
            break
        
        iter += 1

    # 7. Compute and return homogeneous pose matrix Twc.
    Twc = hpose_from_epose(params)
    #------------------

    correct = isinstance(Twc, np.ndarray) and \
        Twc.dtype == np.float64 and \
        Twc.shape == (4, 4) and Twc[3, 3] == 1.0000

    if not correct:
        raise TypeError("Wrong type or size returned!")

    return Twc