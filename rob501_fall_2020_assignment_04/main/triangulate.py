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

    # Compute baseline (right camera translation minus left camera translation).
    cl = np.expand_dims(Twl[:3, 3], axis=1)
    cr = np.expand_dims(Twr[:3, 3], axis=1)
    b = cr - cl
    # Unit vectors projecting from optical center to image plane points.
    # Use variables rayl and rayr for the rays.
    u_rayl = np.vstack((pl, [[1]])) # augmented pixel coordinates
    u_rayl = np.dot(Twl[:3, :3], np.dot(inv(Kl), u_rayl)) # inverse camera model to get world ray coordinate
    rayl = u_rayl/norm(u_rayl) # normalize ray coordinate since its a line

    u_rayr = np.vstack((pr, [[1]])) # repeat for right side
    u_rayr = np.dot(Twr[:3, :3], np.dot(inv(Kr), u_rayr))
    rayr = u_rayr / norm(u_rayr)

    # Projected segment lengths.
    # Use variables ml and mr for the segment lengths.
    r1r2 = np.dot(rayr.T, rayl).item() # Follows Cheng paper to the letter
    ml = ((np.dot(b.T, rayl) - np.dot(b.T, rayr) * r1r2)/(1 - r1r2**2)).item()
    mr = (r1r2 * ml - np.dot(b.T, rayr)).item()
    # Segment endpoints.
    # User variables Pl and Pr for the segment endpoints.
    Pl = cl + rayl*ml # Find endpoint along normalized ray, using computed segment length
    Pr = cr + rayr * mr

    # Now fill in with appropriate ray Jacobians. These are 
    # 3x4 matrices, but two columns are zeros (because the right
    # ray direction is not affected by the left image point and 
    # vice versa).
    drayl = np.zeros((3, 4))  # Jacobian left ray w.r.t. image points.
    drayr = np.zeros((3, 4))  # Jacobian right ray w.r.t. image points

    TKl = np.dot(Twl[:3, :3], inv(Kl))
    TKr = np.dot(Twr[:3, :3], inv(Kr))
    # Jacobian computed for normalized ray vector with respect to pixel coordinates. Computed using basic
    # quotient rule and chain rule
    drayl[:, :2] = (TKl * norm(u_rayl) - np.dot(u_rayl, np.dot(rayl.T, TKl)))[:, :2] / (norm(u_rayl))**2
    drayr[:, 2:] = (TKr * norm(u_rayr) - np.dot(u_rayr, np.dot(rayr.T, TKr)))[:, :2] / (norm(u_rayr))**2

    # Compute dml and dmr (partials wrt segment lengths).
    u = np.dot(b.T, rayl) - np.dot(b.T, rayr)*np.dot(rayl.T, rayr)
    v = 1 - np.dot(rayl.T, rayr)**2

    du = (b.T@drayl).reshape(1, 4) - (b.T@drayr).reshape(1, 4)*np.dot(rayl.T, rayr) - \
         np.dot(b.T, rayr)*((rayr.T@drayl) + (rayl.T@drayr)).reshape(1, 4)
 
    dv = -2*np.dot(rayl.T, rayr)*((rayr.T@drayl).reshape(1, 4) + (rayl.T@drayr).reshape(1, 4))

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

    # 3D point
    P = 0.5*(Pl + Pr) # Midpoint of the line between the right side ray and the left side ray
    # 3x3 landmark point covariance matrix (need to form
    # the 4x4 image plane covariance matrix first).
    Slr = np.zeros((4,4))
    Slr[:2, :2] = Sl
    Slr[2:, 2:] = Sr
    S = np.dot(JP, np.dot(Slr, JP.T)) # Covariance matrix calculated following the procedure from Cheng paper

    # Check for correct outputs...
    correct = isinstance(Pl, np.ndarray) and Pl.shape == (3, 1) and \
              isinstance(Pr, np.ndarray) and Pr.shape == (3, 1) and \
              isinstance(P,  np.ndarray) and P.shape  == (3, 1) and \
              isinstance(S,  np.ndarray) and S.shape  == (3, 3)

    if not correct:
        raise TypeError("Wrong type or size returned!")

    return Pl, Pr, P, S