import numpy as np
from matplotlib import pyplot as plt
import torch


def add_channel_first(array):
    '''
    Adds a new dimension to the given array in the first position
    Example: from (480, 640) to (1, 480, 640)
    Input: array type tensor
    Output: array type tensor with an added dimension
    '''
    assert torch.is_tensor(array), 'array must be a tensor'
    return array.unsqueeze(1)


def add_channel_last(array):
    '''
    Adds a new dimension to the given array in the last position
    Example: from (480, 640) to (480, 640, 1)
    Input: array type tensor
    Output: array type tensor with an added dimension
    '''
    assert torch.is_tensor(array), 'array must be a tensor'
    return array.unsqueeze(2)


def concatenate_first_channel(rgb, depth):
    '''
    Combines two 3D arrays by the first channel
    Example: 
        Input: (3, 480, 640) and (1, 480, 640)
        Output: (4, 480, 640)
    Input: two 3D arrays with type tensor
    Output: one 3D array with type tensor concatenated
    '''
    assert torch.is_tensor(rgb), 'rgb must be a tensor'
    assert torch.is_tensor(depth), 'depth must be a tensor'
    assert rgb.ndim == 3, 'rgb array must have 3 dimensions'
    assert depth.ndim == 3, 'depth array must have 3 dimensions'

    return torch.cat((rgb, depth), dim=0)


def concatenate_last_channel(rgb, depth):
    '''
    THIS FUNCTION IS UNTESTED
    Combines two 3D arrays by the last channel
    Example: 
        Input: (480, 640, 3) and (480, 640, 1)
        Output: (480, 640, 4)
    Input: two 3D arrays with type tensor
    Output: one 3D array with type tensor concatenated
    '''
    assert type(rgb) == np.ndarray, 'rgb array must be a numpy array'
    assert type(depth) == np.ndarray, 'depth array must be a numpy array'
    assert rgb.ndim == 3, 'rgb array must have 3 dimensions'
    assert depth.ndim == 3, 'depth array must have 3 dimensions'

    return np.concatenate((rgb, depth), axis=2)


def split_rgb_depth(array):
    return array[:, :, :3], array[:, :, 3]


def split_rgb_depth_2(array):
    return array[:3, :, :], array[3, :, :]


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)
