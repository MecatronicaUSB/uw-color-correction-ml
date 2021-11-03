import sys
import numpy as np
import unittest
import json
import torch
from torch.utils.data import DataLoader
import copy
import os

sys.path.insert(1, '../../')
from parsers import SeaThruDataset
from utils import np_utils

class TestSeaThruDataset(unittest.TestCase):
    parameters_path = '/../../parameters.json'

    if __name__ == '__main__':
        parameters_path = '../../parameters.json'

    with open(os.path.dirname(__file__) + parameters_path) as path_file:
        params = json.load(path_file)
    
    
    def test_len_overwrite(self):
        dataset = SeaThruDataset(self.params['datasets']['sea-thru-d1'], img_size=(480, 640))
        direct_variable = dataset.length
        len_variable = len(dataset)

        assert direct_variable == len_variable, 'Lengths must match'


    def test_getitem_overwrite(self):
        dataset = SeaThruDataset(self.params['datasets']['sea-thru-d1'], img_size=(480, 640))
        image = dataset[0]
        image_two = dataset.__getitem__(0)

        assert torch.is_tensor(image), 'dataset[0] is not returning a tensor'
        assert torch.is_tensor(image_two), '__getitem__(0) is not returning tensor'
        assert torch.equal(image, image_two), 'Getted items must match'


    def test_get_length(self):
        dataset = SeaThruDataset(self.params['datasets']['sea-thru-d1'], img_size=(480, 640))
        assert len(dataset) == 909, '__len__ must be 909'
    

    def test_get_item(self):
        dataset = SeaThruDataset(self.params['datasets']['sea-thru-d1'], img_size=(480, 640))
        image = dataset[0]

        assert torch.is_tensor(image), 'dataset[0] is not returning a tensor'
        assert image.shape == (3, 480, 640), '__getitem__(0) is not returning original size (3, 480, 640)'


    # def test_plot_image(self):
    #     dataset = SeaThruDataset(self.params['datasets']['sea-thru-d1'])
    #     image = dataset[0].numpy()
    #     image = np_utils.transpose_cwh_to_whc(image)
    #     print(image.shape)
    #     print(image.astype(np.uint8))
    #     np_utils.plot(image.astype(np.uint8))


if __name__ == '__main__':
    unittest.main()