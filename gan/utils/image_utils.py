from PIL import Image
import numpy as np


def using_pil_and_shrink(path, size):
    with Image.open(path) as image:
        # image.draft('RGB', size)
        return np.asarray(image)
