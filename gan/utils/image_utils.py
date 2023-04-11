from PIL import Image
import numpy as np


def usingPILandShrink(path, size):
    with Image.open(path) as image:
        # image.draft('RGB', size)
        return np.asarray(image)
