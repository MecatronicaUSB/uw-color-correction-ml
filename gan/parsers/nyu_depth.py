from utils import np_utils
import h5py
import numpy as np
from torch.utils.data import Dataset
import torch
import sys

sys.path.insert(1, "../")

# ------------------------------------
# -------- Reading a h5 file. --------
# ------------------------------------

# mat_file = h5py.File(dataset_path,'r') ['accelData', 'depths', 'images', 'labels', 'names', 'namesToIds', 'depths', 'scenes']
# data = mat_file.get('images')         # <HDF5 dataset "images": shape (2284, 3, 640, 480), type "|u1">
# depth = mat_file.get('depths')     # <HDF5 dataset "depths": shape (2284, 640, 480), type "<u2">


class NYUDepthParser(Dataset):
    def __init__(self, path):
        super(Dataset, self).__init__()

        assert type(path) == str, "Dataset path must be a string"

        self.dataset = h5py.File(path, "r")

        assert "images" in self.dataset, "Dataset must contain images key"
        assert "depths" in self.dataset, "Dataset must contain depths key"

        self.length = len(self.dataset["images"])
        depth_length = len(self.dataset["depths"])

        assert self.length > 0, "The length of the dataset must be greater than 0"
        assert self.length == depth_length, "Length of images and depths must match"

        print("NYUDepthDataset: length: ", self.length)

        self.CROPPING_PIXELS = 16

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        # ----- Get RGB image and depth image from the dataset
        rgb = np.array(self.dataset["images"][index % self.length])
        depth = np.array(self.dataset["depths"][index % self.length])

        # ----- Transform from (640, 480) to (1, 640, 480)
        depth = np_utils.add_channel_first(depth)

        # ----- Crop pixels at the border of both images
        rgb = rgb[
            :,
            self.CROPPING_PIXELS : -self.CROPPING_PIXELS,
            self.CROPPING_PIXELS : -self.CROPPING_PIXELS,
        ]
        depth = depth[
            :,
            self.CROPPING_PIXELS : -self.CROPPING_PIXELS,
            self.CROPPING_PIXELS : -self.CROPPING_PIXELS,
        ]

        # ----- Transform (x, 640, 480) to (x, 480, 640)
        rgb = np_utils.transpose_cwh_to_chw(rgb)
        depth = np_utils.transpose_cwh_to_chw(depth)

        # ----- Combine images into one with 4 channels
        combined = np_utils.concatenate_first_channel(rgb, depth)
        combined = torch.from_numpy(combined).float()

        return combined
