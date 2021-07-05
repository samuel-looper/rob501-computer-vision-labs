import numpy as np
from numpy.linalg import inv


def rpy_from_dcm(R):
    """
    Roll, pitch, yaw Euler angles from rotation matrix.

    The function computes roll, pitch and yaw angles from the
    rotation matrix R. The pitch angle p is constrained to the range
    (-pi/2, pi/2].  The returned angles are in radians.

    Inputs:
    -------
    R  - 3x3 orthonormal rotation matrix.

    Returns:
    --------
    rpy  - 3x1 np.array of roll, pitch, yaw Euler angles.
    """
    rpy = np.zeros((3, 1))

    # Roll.
    rpy[0] = np.arctan2(R[2, 1], R[2, 2])

    # Pitch.
    sp = -R[2, 0]
    cp = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    if np.abs(cp) > 1e-15:
        rpy[1] = np.arctan2(sp, cp)
    else:
        # Gimbal lock...
        rpy[1] = np.pi / 2

        if sp < 0:
            rpy[1] = -rpy[1]

    # Yaw.
    rpy[2] = np.arctan2(R[1, 0], R[0, 0])

    return rpy

def dcm_from_rpy(rpy):
    """
    Rotation matrix from roll, pitch, yaw Euler angles.

    The function produces a 3x3 orthonormal rotation matrix R
    from the vector rpy containing roll angle r, pitch angle p, and yaw angle
    y.  All angles are specified in radians.  We use the aerospace convention
    here (see descriptions below).  Note that roll, pitch and yaw angles are
    also often denoted by phi, theta, and psi (respectively).

    The angles are applied in the following order:

     1.  Yaw   -> by angle 'y' in the local (body-attached) frame.
     2.  Pitch -> by angle 'p' in the local frame.
     3.  Roll  -> by angle 'r' in the local frame.

    Note that this is exactly equivalent to the following fixed-axis
    sequence:

     1.  Roll  -> by angle 'r' in the fixed frame.
     2.  Pitch -> by angle 'p' in the fixed frame.
     3.  Yaw   -> by angle 'y' in the fixed frame.

    Parameters:
    -----------
    rpy  - 3x1 np.array of roll, pitch, yaw Euler angles.

    Returns:
    --------
    R  - 3x3 np.array, orthonormal rotation matrix.
    """
    cr = np.cos(rpy[0]).item()
    sr = np.sin(rpy[0]).item()
    cp = np.cos(rpy[1]).item()
    sp = np.sin(rpy[1]).item()
    cy = np.cos(rpy[2]).item()
    sy = np.sin(rpy[2]).item()

    return np.array([[cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
                     [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
                     [  -sp,            cp*sr,            cp*cr]])

def find_jacobian(K, Twc, Wpt):
    """
    Determine the Jacobian for NLS camera pose optimization.

    The function computes the Jacobian of an image plane point with respect
    to the current camera pose estimate, given a landmark point. The 
    projection model is the simple pinhole model.

    Note that the homogeneous transformation matrix provided defines the
    transformation from the *camera frame* to the *world frame* (to 
    project into the image, you would need to invert this matrix).

    Parameters:
    -----------
    K    - 3x3 np.array, camera intrinsic calibration matrix.
    Twc  - 4x4 np.array, homogenous pose matrix, current guess for camera pose. 
    Wpt  - 3x1 world point on calibration target (one of n).

    Returns:
    --------
    J  - 2x6 np.array, Jacobian matrix (columns are tx, ty, tz, r, p, q).
         The Jacobian must contain float64 values.
    """

    # Calculate first Jacobian: from homogenous representation to image coordinate representation
    w_hom = np.vstack([Wpt, [1]])
    p_c = np.dot(inv(Twc), w_hom)
    x_tilde = np.dot(K, p_c[:3, :])

    first_jac = np.asarray([[1/x_tilde[2, 0], 0, -x_tilde[0, 0]/(x_tilde[2, 0] ** 2)],
                            [0, 1/x_tilde[2, 0], -x_tilde[1, 0]/(x_tilde[2, 0] ** 2)]])

    # Calculate Third Jacobian: from World homogenous representation to Camera parameters (tx, ty, tz, r, p, y)
    Cwc = (Twc)[0:3, 0:3]
    trans = np.asarray([Twc[:3, 3]]).T
    angs = rpy_from_dcm(Cwc) # Calculate Euler Angles
    r = angs[0, 0]
    p = angs[1, 0]
    y = angs[2, 0]

    # Calculate partial Jacobian of rotation matrix w.r.t Euler Angles
    C_3 = inv(dcm_from_rpy(np.asarray([[r, 0, 0]]).T))
    C_2 = inv(dcm_from_rpy(np.asarray([[0, p, 0]]).T))
    C_1 = inv(dcm_from_rpy(np.asarray([[0, 0, y]]).T))
    cx_3 = np.asarray([[0, 0, 0],
                       [0, 0, -1],
                       [0, 1, 0]])
    cx_2 = np.asarray([[0, 0, 1],
                       [0, 0, 0],
                       [-1, 0, 0]])
    cx_1 = np.asarray([[0, -1, 0],
                       [1, 0, 0],
                       [0, 0, 0]])
    Jc_3 = np.dot(cx_3, np.dot(C_3, np.dot(C_2, np.dot(C_1, Wpt-trans))))
    Jc_2 = np.dot(C_3, np.dot(cx_2, np.dot(C_2, np.dot(C_1, Wpt-trans))))
    Jc_1 = np.dot(C_3, np.dot(C_2, np.dot(cx_1, np.dot(C_1, Wpt-trans))))
    third_jac = np.hstack([-Cwc.T, -Jc_3, -Jc_2, -Jc_1]) # Third Jacobian calculation for 6 parameters

    # Calculate Jacobian by chain rule
    J = np.dot(first_jac, np.dot(K, third_jac))

    #------------------

    correct = isinstance(J, np.ndarray) and \
        J.dtype == np.float64 and J.shape == (2, 6)

    if not correct:
        raise TypeError("Wrong type or size returned!")

    return J