
import numexpr as ne
import numpy as np


def get_2d_gaussian(shape, sigma=1):
    # type: ((int, int), float) -> np.ndarray
    """
    :param shape: output map shape
    :param sigma: Gaussian standard dev.
    :return: map with a Gaussian in the center
    """
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]

    local_dict = {'x': x, 'y': y, 'sigma': sigma}
    h = ne.evaluate('exp(-(x**2 + y**2) / (2* (sigma**2)))', local_dict=local_dict)
    # alternative (with standard numpy): h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_gaussian(heatmap, center, sigma, peak_value=1):
    # type: (np.ndarray, (int, int), float, float) -> None
    """
    Draw a Gaussian (on `heatmap`) centered in `center` with the required sigma value.
    NOTE: inplace function --> `heatmap` will be modified!

    :param heatmap: map on which to draw the Gaussian; shape (H, W)
    :param center: Gaussian center
    :param sigma: Gaussian standard deviation
    :param peak_value: Gaussian peak value (default 1)
    """
    diameter = sigma * 6 + 1
    radius = (diameter - 1) // 2
    gaussian = get_2d_gaussian((diameter, diameter), sigma=sigma)

    x, y = center
    height, width = heatmap.shape[0:2]
    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    np.maximum(masked_heatmap, masked_gaussian * peak_value, out=masked_heatmap)


# hmap = np.zeros((256, 256))
# draw_gaussian(hmap, (40, 100), sigma=6)
