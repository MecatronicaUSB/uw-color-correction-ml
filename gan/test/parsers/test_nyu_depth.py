import sys
import numpy as np
import unittest
import json
import torch
from torch.utils.data import DataLoader
import copy
import os
from torchvision.utils import save_image

sys.path.insert(1, '../../')
from parsers import NYUDepthDataset
from utils import np_utils

class TestNYUDepthDataset(unittest.TestCase):
    parameters_path = '/../../parameters.json'

    if __name__ == '__main__':
        parameters_path = '../../parameters.json'

    with open(os.path.dirname(__file__) + parameters_path) as path_file:
        params = json.load(path_file)
    

    def test_len_overwrite(self):
        dataset = NYUDepthDataset(self.params['datasets']['nyu-depth-v2'])
        direct_variable = dataset.length
        len_variable = len(dataset)

        assert direct_variable == len_variable, 'Lengths must match'


    def test_getitem_overwrite(self):
        dataset = NYUDepthDataset(self.params['datasets']['nyu-depth-v2'])
        image = dataset[0]
        image_two = dataset.__getitem__(0)

        assert torch.is_tensor(image), 'dataset[0] is not returning a tensor'
        assert torch.is_tensor(image_two), '__getitem__(0) is not returning tensor'
        assert torch.equal(image, image_two), 'Getted items must match'


    def test_get_length(self):
        dataset = NYUDepthDataset(self.params['datasets']['nyu-depth-v2'])
        assert len(dataset) == 1449, '__len__ must be 1449'
    

    def test_get_item(self):
        dataset = NYUDepthDataset(self.params['datasets']['nyu-depth-v2'])
        image = dataset[0]

        assert torch.is_tensor(image), 'dataset[0] is not returning a tensor'
        assert image.shape == (4, 480, 640), '__getitem__(0) is not returning original size (4, 480, 640)'


    # def test_plot_image(self):
    #     dataset = NYUDepthDataset(self.params['datasets']['nyu-depth-v2'])
    #     image = dataset[0][:3, :, :]
    #     save_image(image, "images/%d.png" % 36, nrow=5, normalize=True)

    #     image = dataset[0].numpy()[:3, :, :]
    #     print(image.shape)
    #     image = np_utils.transpose_cwh_to_whc(image)
    #     print(image.shape)
    #     print(image.astype(np.uint8))
    #     np_utils.plot(image.astype(np.uint8))


if __name__ == '__main__':
    unittest.main()