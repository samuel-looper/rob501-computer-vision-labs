import numpy as np
from numpy.linalg import inv

def bilinear_interp(I, pt):
    """
    Performs bilinear interpolation for a given image point.

    Given the (x, y) location of a point in an input image, use the surrounding
    4 pixels to conmpute the bilinearly-interpolated output pixel intensity.

    Note that images are (usually) integer-valued functions (in 2D), therefore
    the intensity value you return must be an integer (use round()).

    This function is for a *single* image band only - for RGB images, you will 
    need to call the function once for each colour channel.

    Parameters:
    -----------
    I   - Single-band (greyscale) intensity image, 8-bit np.array (i.e., uint8).
    pt  - 2x1 np.array of point in input image (x, y), with subpixel precision.

    Returns:
    --------
    b  - Interpolated brightness or intensity value (whole number >= 0).
    """
    if pt.shape != (2, 1):
        raise ValueError('Point size is incorrect.')

    pt = np.squeeze(pt, axis=1)
    # print(pt)
    pt1 = np.floor(pt)
    pt2 = np.floor(pt)+1
    x1 = int(np.clip(pt1[0], 0, I.shape[1]-1))
    y1 = int(np.clip(pt1[1], 0, I.shape[0]-1))
    x2 = int(np.clip(pt2[0], 0, I.shape[1]-1))
    y2 = int(np.clip(pt2[1], 0, I.shape[0]-1))

    adj_pts = np.array([[x1, x1, x2, x2], [y1, y2, y1, y2]])
    # print(adj_pts[1,:])

    # Extract corresponding intensities
    val_vec = I[adj_pts[1, :], adj_pts[0, :]]
    w = np.array([(x2 - pt[0]) * (y2 - pt[1]), (x2 - pt[0]) * (pt[1] - y1),
                  (pt[0] - x1) * (y2 - pt[1]), (pt[0] - x1) * (pt[1] - y1)], dtype=np.float32)

    b = np.round(np.dot(val_vec, w)).astype(np.uint8)
    #------------------

    return b
