from utils import np_utils, using_pil_and_shrink
import numpy as np
import torch
import sys
import copy
from PIL import Image
from torch.utils.data import Dataset
from os import listdir
from os.path import isfile, join

sys.path.insert(1, '../')


class UWDataset(Dataset):
    def __init__(self, images_path, gt_path, img_size=(480, 640)):
        super(Dataset, self).__init__()

        assert type(images_path) == str, 'Dataset path must be a string'
        assert type(gt_path) == str, 'Dataset path must be a string'

        # ---- Get the name of all files in this directory
        self.images_path = [join(images_path, f) for f in listdir(
            images_path) if isfile(join(images_path, f))]
        self.gt_path = [join(gt_path, f)
                        for f in listdir(gt_path) if isfile(join(gt_path, f))]

        self.length = len(self.images_path)
        self.img_size = img_size

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        # ---- Load image and resize
        image = using_pil_and_shrink(
            self.images_path[index % self.length], self.img_size)
        gt = using_pil_and_shrink(
            self.gt_path[index % self.length], self.img_size)

        # ---- To numpy
        image = np.asarray(image)
        gt = np.asarray(gt)

        # ---- From (480, 640, 3) to (3, 480, 640)
        image = np_utils.transpose_hwc_to_chw(image)
        gt = np_utils.transpose_hwc_to_chw(gt)

        # ---- To torch
        image = torch.from_numpy(copy.deepcopy(image)).float()
        gt = torch.from_numpy(copy.deepcopy(gt)).float()

        return image, gt
