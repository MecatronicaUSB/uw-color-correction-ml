from utils import np_utils, using_pil_and_shrink
import numpy as np
import torch
import sys
import copy
from torch.utils.data import Dataset
from os import listdir
from os.path import isfile, join

sys.path.insert(1, "../")


class SeaThruParser(Dataset):
    def __init__(self, path, img_size=(448, 608)):
        super(Dataset, self).__init__()

        assert type(path) == str, "Dataset path must be a string"

        # Get the name of all files in this directory
        self.files_paths = [
            join(path, f) for f in listdir(path) if isfile(join(path, f))
        ]

        self.length = len(self.files_paths)
        self.img_size = img_size
        print("SeaThruDataset: length: ", self.length)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        # TODO: improvement idea: we can randomize this index so we don't get
        # the same in-air match all the time

        # Load image and resize
        image = using_pil_and_shrink(self.files_paths[index % self.length], None)

        # To numpy
        image = np.asarray(image)

        # ---- From (480, 640, 3) to (3, 480, 640)
        image = np_utils.transpose_hwc_to_chw(image)

        # To torch
        image = torch.from_numpy(copy.deepcopy(image)).float()

        return image
