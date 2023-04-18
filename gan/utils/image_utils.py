from PIL import Image
import numpy as np
import torch
import copy
from . import np_utils


def using_pil_and_shrink(path, size=None):
    with Image.open(path) as image:
        if size is not None:
            image = image.resize(size)
        # image.draft('RGB', size)
        return np.asarray(image)


def load_image_to_eval(path, device, size=None):
    image = np.asarray(using_pil_and_shrink(path, size))
    image = np_utils.transpose_hwc_to_chw(image)
    image = np_utils.add_channel_first(image)
    image = torch.from_numpy(copy.deepcopy(image)).float().to(device) / 255

    return image
