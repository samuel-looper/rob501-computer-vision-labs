import numpy as np

def alpha_blend(Ifg, Ibg, alpha):
    """
    Performs alpha blending of two images.

    The alpha parameter controls the amount of blending between the foreground
    image and the background image. The background image is assumed to be fully
    opaque.

    Note that images are (usually) integer-valued functions, therefore the 
    image you return must be integer-valued (use round()).

    This function should accept both greyscale and RGB images, and return an
    image of the same type.

    Parameters:
    -----------
    Ifg    - Greyscale or RGB foreground image, 8-bit np.array (i.e., uint8).
    Ibg    - Greyscale or RGB background image, 8-bit np.array (i.e., uint8).
    alpha  - Floating point blending parameter, [0, 1].

    Returns:
    --------
    Ia  - Alpha-bended image (*must be* same size as original images).
    """

    if Ifg.shape != Ibg.shape:
        raise ValueError('Input images are different sizes.')

    Ia = np.round((Ifg * alpha) + Ibg * (1-alpha)).astype(np.uint8)


    return Ia
