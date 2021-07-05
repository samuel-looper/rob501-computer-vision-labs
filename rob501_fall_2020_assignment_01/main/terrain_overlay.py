# Terrain overlay script file.
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from imageio import imread, imwrite

from dlt_homography import dlt_homography
from bilinear_interp import bilinear_interp
from alpha_blend import alpha_blend
import timeit


def terrain_overlay(Ifg = None, Ibg = None,
                    Ifg_pts = None, Ibg_pts = None, bbox = None):
    """
    Help rover drivers to safely navigate around obstacles by producing a
    terrain overlay!

    Returns:
    --------
    Iover  - Terrain overlay RGB image, 8-bit np.array (i.e., uint8).
    """
    if not Ifg:
      # Bounding box in rover navcam image (if useful).
      bbox = np.array([[62, 1242, 1242, 62], [190, 190, 794, 794]])

      # Point correspondences.
      Ibg_pts = np.array([[410, 928, 1240, 64], [192,  192, 792, 786]])
      Ifg_pts = np.array([[2, 898, 898, 2], [2, 2, 601, 601]])

      Ibg = imread('../images/rover_forward_navcam.png')
      Ifg = imread('../images/false_colour_overlay.png')

    (H, A) = dlt_homography(Ibg_pts, Ifg_pts)  # Matrix to transform point from background to foreground

    Iover = Ibg.copy()
    Ifg = np.asarray(Ifg)
    Ibg = np.asarray(Ibg)
    alpha = 0.7

    I_warped = Iover
    coords = np.indices((Iover[:, :, 0]).shape)
    one_mat = np.expand_dims(np.ones((Iover[:, :, 0]).shape), axis=0)
    coords = np.vstack([np.expand_dims(coords[1,:], axis=0), np.expand_dims(coords[0,:], axis=0), one_mat])
    coords = np.reshape(coords, (3, coords.shape[1]*coords.shape[2])).astype(np.uint32)
    bbox_path = Path(bbox.T)
    coords = coords[:, bbox_path.contains_points(coords[:2, :].T)].astype(np.uint32)
    warp_coord = np.dot(H, coords).astype(np.float64)
    warp_coord /= warp_coord[2, :]

    for i in range(warp_coord.shape[1]):
        # if i % 100000 == 0:
        #     print(i)
        if 0 <= warp_coord[0, i] < Ifg.shape[1]-1 and 0 <= warp_coord[1, i] < Ifg.shape[0]-1:
            a = bilinear_interp(Ifg[:, :, 0], np.asarray(warp_coord[:2, i]).reshape(2,1))
            b = bilinear_interp(Ifg[:, :, 1], np.asarray(warp_coord[:2, i]).reshape(2,1))
            c = bilinear_interp(Ifg[:, :, 2], np.asarray(warp_coord[:2, i]).reshape(2,1))
            I_warped[coords[1, i], coords[0, i], :] = np.array([a, b, c])



    plt.imshow(I_warped)
    plt.show()
    Iover = alpha_blend(I_warped, Ibg, alpha)

    plt.imshow(Iover)
    plt.show()
    imwrite('terrain_overlay.png', Iover)

    return Iover

if __name__ == "__main__":

    Iover = terrain_overlay()