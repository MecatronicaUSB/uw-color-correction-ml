import numpy as np
from matplotlib import pyplot as plt
import torch

def add_channel_first(array):
    '''
    Adds a new dimension to the given array in the first position
    Example: from (480, 640) to (1, 480, 640)
    Input: array type np.ndarray
    Output: array type np.ndarray with an added dimension
    '''
    assert type(array) == np.ndarray, 'array must be a numpy array'
    return array[np.newaxis, ...]

def add_channel_second(array):
    '''
    Adds a new dimension to the given array in the first position
    Example: from (480, 640) to (1, 480, 640)
    Input: array type np.ndarray
    Output: array type np.ndarray with an added dimension
    '''
    assert type(array) == np.ndarray, 'array must be a numpy array'
    return array[:, np.newaxis, ...]


def add_channel_last(array):
    '''
    Adds a new dimension to the given array in the last position
    Example: from (480, 640) to (480, 640, 1)
    Input: array type np.ndarray
    Output: array type np.ndarray with an added dimension
    '''
    assert type(array) == np.ndarray, 'array must be a numpy array'
    return array[..., np.newaxis]

def concatenate_first_channel(rgb, depth):
    '''
    Combines two 3D arrays by the first channel
    Example: 
        Input: (3, 480, 640) and (1, 480, 640)
        Output: (4, 480, 640)
    Input: two 3D arrays with type np.ndarray
    Output: one 3D array with type np.ndarray concatenated
    '''
    assert type(rgb) == np.ndarray, 'rgb array must be a numpy array'
    assert type(depth) == np.ndarray, 'depth array must be a numpy array'
    assert rgb.ndim == 3, 'rgb array must have 3 dimensions'
    assert depth.ndim == 3, 'depth array must have 3 dimensions'

    return np.concatenate((rgb, depth), axis=0)

def concatenate_last_channel(rgb, depth):
    '''
    THIS FUNCTION IS UNTESTED
    Combines two 3D arrays by the last channel
    Example: 
        Input: (480, 640, 3) and (480, 640, 1)
        Output: (480, 640, 4)
    Input: two 3D arrays with type np.ndarray
    Output: one 3D array with type np.ndarray concatenated
    '''
    assert type(rgb) == np.ndarray, 'rgb array must be a numpy array'
    assert type(depth) == np.ndarray, 'depth array must be a numpy array'
    assert rgb.ndim == 3, 'rgb array must have 3 dimensions'
    assert depth.ndim == 3, 'depth array must have 3 dimensions'
    
    return np.concatenate((rgb, depth), axis=2)

def transpose_cwh_to_whc(array):
    '''
    Transpose an array with dimensions (c, w, h) to (w, h, c)
    Example: 
        Input: (3, 480, 640)
        Output: (480, 640, 3)
    Input: a 3D array with type np.ndarray
    Output: one 3D array with type np.ndarray transposed
    '''
    assert type(array) == np.ndarray, 'array must be a numpy array'
    assert array.ndim == 3, 'array must have 3 dimensions'
    
    return np.transpose(array, (2, 1, 0))

def transpose_cwh_to_chw(array):
    assert type(array) == np.ndarray, 'array must be a numpy array'
    assert array.ndim == 3, 'array must have 3 dimensions'

    return np.transpose(array, (0, 2, 1))

def transpose_bcwh_to_bchw(array):
    assert type(array) == np.ndarray, 'array must be a numpy array'
    assert array.ndim == 4, 'array must have 4 dimensions'

    return np.transpose(array, (0, 1, 3, 2))

def transpose_cwh_to_hwc(array):
    '''
    Transpose an array with dimensions (c, w, h) to (h, w, c)
    Example: 
        Input: (3, 480, 640)
        Output: (480, 640, 3)
    Input: a 3D array with type np.ndarray
    Output: one 3D array with type np.ndarray transposed
    '''
    assert type(array) == np.ndarray, 'array must be a numpy array'
    assert array.ndim == 3, 'array must have 3 dimensions'
    
    return np.transpose(array, (2, 1, 0))

def transpose_hwc_to_chw(array):
    '''
    Transpose an array with dimensions (h, w, c) to (c, h, w)
    Example: 
        Input: (480, 640, 3)
        Output: (3, 480, 640)
    Input: a 3D array with type np.ndarray
    Output: one 3D array with type np.ndarray transposed
    '''
    assert type(array) == np.ndarray, 'array must be a numpy array'
    assert array.ndim == 3, 'array must have 3 dimensions'
    
    return np.transpose(array, (2, 0, 1))

def split_rgb_depth(array):
    return array[:,:,:3], array[:,:,3]

def split_rgb_depth_2(array):
    return array[:3,:,:], array[3,:,:]

def plot(image):
    plt.imshow(image)
    plt.show()

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)