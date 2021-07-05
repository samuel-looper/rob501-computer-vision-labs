import numpy as np
from scipy.ndimage.filters import *

def stereo_disparity_best(Il, Ir, bbox, maxd):
    """
    Best stereo correspondence algorithm.

    This function computes a stereo disparity image from left stereo 
    image Il and right stereo image Ir. Only disparity values within
    the bounding box region (inclusive) are evaluated.

    Parameters:
    -----------
    Il    - Left stereo image, m x n pixel np.array, greyscale.
    Ir    - Right stereo image, m x n pixel np.array, greyscale.
    bbox  - 2x2 np.array, bounding box, relative to left image, from top left
            corner to bottom right corner (inclusive). (x, y) in columns.
    maxd  - Integer, maximum disparity value; disparities must be within zero
            to maxd inclusive (i.e., don't search beyond maxd).

    Returns:
    --------
    Id  - Disparity image (map) as np.array, same size as Il.
    """

    #--- FILL ME IN ---
    # Unfortunately I am allergic to extra work and particularly busy this week so I slapped a median filter on my
    # local method and used hyperparameter optimization to find the optimal local window size and median filter size.

    filter_size = 6             # Optimal local window size
    b1 = int(filter_size / 2)   # Integer halves to ensure local window is centered at the chosen coordinate
    b2 = filter_size - b1
    x_lb = bbox[0, 0]           # renaming bounding box values for the aesthetic
    y_lb = bbox[1, 0]
    x_rb = bbox[0, 1]
    y_rb = bbox[1, 1]
    x_end = Il.shape[1]
    y_end = Il.shape[0]

    # Create mirror-padded image (Left)
    Il_padded = np.zeros((Il.shape[0] + filter_size, Il.shape[1] + filter_size))
    Il_padded[:b1, b1:x_end + b1] = np.expand_dims(Il[0, :], axis=0)
    Il_padded[b1:y_end + b1, :b1] = np.expand_dims(Il[:, 0], axis=1)
    Il_padded[y_end + b1:y_end + filter_size, b1:x_end + b1] = np.expand_dims(Il[-1, :], axis=0)
    Il_padded[b1:y_end + b1, x_end + b1:x_end + filter_size] = np.expand_dims(Il[:, -1], axis=1)
    Il_padded[:b1, :b1] = np.ones((b1, b1)) * Il[0, 0]
    Il_padded[y_end + b1:y_end + filter_size, :b1] = np.ones((b2, b1)) * Il[-1, 0]
    Il_padded[:b1, x_end + b1:x_end + filter_size] = np.ones((b1, b2)) * Il[0, -1]
    Il_padded[y_end + b1:y_end + filter_size, x_end + b1:x_end + filter_size] = np.ones((b2, b2)) * Il[-1, -1]
    Il_padded[b1:b1 + y_end, b1:b1 + x_end] = Il

    # Create mirror-padded image (Right)
    Ir_padded = np.zeros((Ir.shape[0] + filter_size, Ir.shape[1] + filter_size))
    Ir_padded[:b1, b1:x_end + b1] = np.expand_dims(Ir[0, :], axis=0)
    Ir_padded[b1:y_end + b1, :b1] = np.expand_dims(Ir[:, 0], axis=1)
    Ir_padded[y_end + b1:y_end + filter_size, b1:x_end + b1] = np.expand_dims(Ir[-1, :], axis=0)
    Ir_padded[b1:y_end + b1, x_end + b1:x_end + filter_size] = np.expand_dims(Ir[:, -1], axis=1)
    Ir_padded[:b1, :b1] = np.ones((b1, b1)) * Ir[0, 0]
    Ir_padded[y_end + b1:y_end + filter_size, :b1] = np.ones((b2, b1)) * Ir[-1, 0]
    Ir_padded[:b1, x_end + b1:x_end + filter_size] = np.ones((b1, b2)) * Ir[0, -1]
    Ir_padded[y_end + b1:y_end + filter_size, x_end + b1:x_end + filter_size] = np.ones((b2, b2)) * Ir[-1, -1]
    Ir_padded[b1:b1 + y_end, b1:b1 + x_end] = Ir

    Id = np.zeros(Il.shape) # Initialize result image
    # Iterate over all points in bounding box, not including padding pixels
    for i in range(x_lb + b1, x_rb):
        for j in range(y_lb + b1, y_rb):
            # Get local window filter to compare left and right images
            im_filter = Il_padded[j - b1:j + b2, i - b1:i + b2]
            min_sad = 10000
            min_ind = i
            for k in range(max(i - maxd, b1), i): # Iterate over all previous x values
                # Calculate Sum of Absolute Differences for a window of the same size centered at new x value
                sad = np.mean(np.abs(np.subtract(im_filter, Ir_padded[j - b1:j + b2, k - b1:k + b2])))
                # Track lowest SAD value and its index
                if min_sad > sad:
                    min_sad = sad
                    min_ind = k
            Id[j, i] = np.abs(i - min_ind)  # Set intensity of result image to disparity at lowest SAD value

    # Padding resulting image for good measure
    Id[:y_lb + b1, x_lb + b1:] = np.expand_dims(Id[y_lb + b1, x_lb + b1:], axis=0)
    Id[y_lb + b1:, :x_lb + b1] = np.expand_dims(Id[y_lb + b1:, x_lb + b1], axis=1)
    Id[:y_lb + b1, :x_lb + b1] = np.ones(Id[:y_lb + b1, :x_lb + b1].shape) * Id[y_lb + b1, x_lb + b1]

    Id[:y_rb, x_rb - 1:] = np.expand_dims(Id[:y_rb, x_rb - 1], axis=1)
    Id[y_rb - 1:, :x_rb] = np.expand_dims(Id[y_rb - 1, :x_rb], axis=0)
    Id[y_rb - 1:, x_rb - 1:] = np.ones(Id[y_rb - 1:, x_rb - 1:].shape) * Id[y_rb - 1, x_rb - 1]

    # Apply magical median filter, filter size obtained via hyperparameter optimization
    Id = median_filter(Id, 30)

    #------------------

    correct = isinstance(Id, np.ndarray) and Id.shape == Il.shape

    if not correct:
        raise TypeError("Wrong type or size returned!")

    return Id