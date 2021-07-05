import numpy as np
from numpy.linalg import inv
from support.rpy_from_dcm import rpy_from_dcm
from support.dcm_from_rpy import dcm_from_rpy
from templates.estimate_motion_ls import estimate_motion_ls


def estimate_motion_ils(Pi, Pf, Si, Sf, iters):
    """
    Estimate motion from 3D correspondences.

    The function estimates the 6-DOF motion of a body, given a series
    of 3D point correspondences. This method relies on NLS.

    Arrays Pi and Pf store corresponding landmark points before and after
    a change in pose.  Covariance matrices for the points are stored in Si
    and Sf. All arrays should contain float64 values.

    Parameters:
    -----------
    Pi  - 3xn np.array of points (intial - before motion).
    Pf  - 3xn np.array of points (final - after motion).
    Si  - 3x3xn np.array of landmark covariance matrices.
    Sf  - 3x3xn np.array of landmark covariance matrices.

    Outputs:
    --------
    Tfi  - 4x4 np.array, homogeneous transform matrix, frame 'i' to frame 'f'.
    """
    # Initial guess...
    Tfi = estimate_motion_ls(Pi, Pf, Si, Sf)
    C = Tfi[:3, :3]
    I = np.eye(3)
    rpy = rpy_from_dcm(C).reshape(3, 1)

    # Iterate
    for j in np.arange(iters):
        A = np.zeros((6, 6))
        B = np.zeros((6, 1))
        Cfi = (Tfi)[0:3, 0:3]                   # Rotation transformation matrix for this iteration
        dRdr, dRdp, dRdy = dcm_jacob_rpy(Cfi)   # Calculate partials with respect to euler angles

        # --- FILL ME IN ---
        for i in np.arange(Pi.shape[1]):
            # Calculate weight matrix based on inverse of the point covariances (uncertainties)
            S_i = Si[:, :, i] + np.dot(C, np.dot(Sf[:, :, i], C.T))

            # Initialize point correspondences
            pf_i = np.expand_dims(Pf[:, i], axis=1) # Final point
            pi_i = np.expand_dims(Pi[:, i], axis=1) # Initial point

            # Calculate Jacobian (based on Matthies Thesis)
            J = np.hstack((np.dot(dRdr, pi_i), np.dot(dRdp, pi_i), np.dot(dRdy, pi_i), I))

            # Calculate residual
            r_i = pf_i - np.dot(Cfi, pi_i) + np.dot(J[:, :3], rpy)

            # Calculate component of A and B matrix and sum over all point correspondences
            A += np.dot(J.T, np.dot(inv(S_i), J))
            B += np.dot(J.T, np.dot(inv(S_i), r_i))
        # ------------------

        # Solve system and check stopping criteria if desired...
        theta = inv(A)@B
        rpy = theta[0:3].reshape(3, 1)
        C = dcm_from_rpy(rpy)
        t = theta[3:6].reshape(3, 1)

    Tfi = np.vstack((np.hstack((C, t)), np.array([[0, 0, 0, 1]])))

    # Check for correct outputs...
    correct = isinstance(Tfi, np.ndarray) and Tfi.shape == (4, 4)

    if not correct:
        raise TypeError("Wrong type or size returned!")

    return Tfi

def dcm_jacob_rpy(C):
     # Rotation - convenient!
    cp = np.sqrt(1 - C[2, 0]*C[2, 0])
    cy = C[0, 0]/cp
    sy = C[1, 0]/cp

    dRdr = C@np.array([[ 0,   0,   0],
                       [ 0,   0,  -1],
                       [ 0,   1,   0]])

    dRdp = np.array([[ 0,    0, cy],
                     [ 0,    0, sy],
                     [-cy, -sy,  0]])@C

    dRdy = np.array([[ 0,  -1,  0],
                     [ 1,   0,  0],
                     [ 0,   0,  0]])@C

    return dRdr, dRdp, dRdy