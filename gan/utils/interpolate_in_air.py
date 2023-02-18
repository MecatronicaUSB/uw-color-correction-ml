import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp2d
import copy
import json
import os


def interpolate_missing_pixels(
        image: np.ndarray,
        mask: np.ndarray,
        method: str = 'nearest',
        fill_value: int = 0
):
    """
    :param image: a 2D image
    :param mask: a 2D boolean image, True indicates missing values
    :param method: interpolation method, one of
        'nearest', 'linear', 'cubic'.
    :param fill_value: which value to use for filling up data outside the
        convex hull of known pixel values.
        Default is 0, Has no effect for 'nearest'.
    :return: the image with missing values interpolated
    """
    from scipy import interpolate

    h, w = image.shape[:2]
    xx, yy = np.meshgrid(np.arange(w), np.arange(h))

    known_x = xx[~mask]
    known_y = yy[~mask]
    known_v = image[~mask]
    missing_x = xx[mask]
    missing_y = yy[mask]

    interp_values = interpolate.griddata(
        (known_x, known_y), known_v, (missing_x, missing_y),
        method=method, fill_value=fill_value
    )

    interp_image = image.copy()
    interp_image[missing_y, missing_x] = interp_values

    return interp_image


def process_image(image, transpose):
    copy_image = copy.deepcopy(image)

    # Creating the padding mask (the border of the img that has depth 0)
    padding_mask = np.zeros((640, 480))
    padding_mask[0:45, :] = 1
    padding_mask[:, 0:45] = 1
    padding_mask[600:640, :] = 1
    padding_mask[:, 470:480] = 1

    # Interpolate 3 times in different ways
    filled_image = interpolate_missing_pixels(
        copy_image, copy_image == 0, 'cubic')
    pad_fixed_image = interpolate_missing_pixels(
        filled_image, padding_mask == 1, 'nearest')
    pad_fixed_image = interpolate_missing_pixels(
        pad_fixed_image, pad_fixed_image == 0, 'nearest')

    # Fixing stuff that could've happened in the interpolation
    pad_fixed_image[pad_fixed_image < 0.5] = 0.5
    pad_fixed_image[pad_fixed_image > 10] = 10

    # So the color map stays the same in matplolib
    # pad_fixed_image[0][0] = 0

    if transpose:
        return np.transpose(pad_fixed_image, (1, 0))
    else:
        return pad_fixed_image


# f, axarr = plt.subplots(2, 2)
# i = 0
# axarr[0, 0].imshow(np.transpose(
#     dataset['rawDepths'][i], (1, 0)), interpolation='None')
# axarr[0, 1].imshow(np.transpose(
#     dataset['depths'][i+1], (1, 0)), interpolation='None')
# axarr[1, 0].imshow(np.transpose(
#     dataset['depths'][i+2], (1, 0)), interpolation='None')
# axarr[1, 1].imshow(np.transpose(
#     dataset['depths'][i+3], (1, 0)), interpolation='None')
# plt.show()

# ---------- Opening parameters
with open(os.path.dirname(__file__) + "/../parameters.json") as path_file:
    params = json.load(path_file)

dataset = h5py.File(params["datasets"]["in-air"], 'r')
output_path = params["datasets"]["in-air"].split("/")[:-1] + "/output.mat"
output_file = h5py.File(params["datasets"]["in-air"], 'w')

for i in range(len(dataset['rawDepths'])):
    print()
