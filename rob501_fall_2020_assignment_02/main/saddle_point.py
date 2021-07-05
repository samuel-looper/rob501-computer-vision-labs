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

    # Building Matrices for Least Squares Optimization
    indx = (np.indices(I.shape))
    ys = np.ravel(indx[0, :, :])
    xs = np.ravel(indx[1, :, :])
    flatten_I = np.ravel(I) # flattened representation of all image point intensity values
    aug_I = np.zeros((I.size, 6)) # Linear representation of quadratic function for optimization
    for i in range(I.size):
        x = xs[i]
        y = ys[i]
        aug_I[i, :] = np.asarray([x**2, y*x, y**2, x, y, 1])

    params, residuals, rank, s = np.linalg.lstsq(aug_I, flatten_I, rcond=None) # Least squares optimization

    # Solution for the saddle point of the best fit quadratic function
    param_mat = np.asarray([[2*params[0], params[1]], [params[1], 2*params[2]]])
    pt = -np.dot(np.linalg.inv(param_mat), np.asarray([[params[3], params[4]]]).T)

    #------------------

    correct = isinstance(pt, np.ndarray) and \
        pt.dtype == np.float64 and pt.shape == (2, 1)

    if not correct:
        raise TypeError("Wrong type or size returned!")

    return pt