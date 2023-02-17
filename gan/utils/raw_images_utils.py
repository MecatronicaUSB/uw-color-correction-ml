import rawpy
from os import listdir
from os.path import isfile, join
from PIL import Image
import numpy as np
import os
import json


def get_rawpy_params():
    """
    Gets rawpy parameters for postprocessing.
    :return: Dictionary of parameters for rawpy.
    """
    def get_value(d, key, default_key):
        """
        Gets a value coresponding to the specified key from the dictionary.
        :param d: Dictionary.
        :param key: Key.
        :param default_key: Default key to use if the one specified is not found.
        :return: Value corresponding to key in the dictionary.
        """
        if key in d:
            return d[key]
        return d[default_key]

    use_camera_wb = False
    use_auto_wb = False
    bright = 1.0
    median_filter_passes = 0
    noise_thr = None
    dcb_enhance = False
    four_color_rgb = False
    demosaic_algorithm = 'AHD'
    fbdd_noise_reduction = 'Off'
    output_color = 'sRGB'
    output_bps = 8

    demosaic_algorithms = {
        'AAHD': rawpy.DemosaicAlgorithm.AAHD,
        'AFD': rawpy.DemosaicAlgorithm.AFD,
        'AHD': rawpy.DemosaicAlgorithm.AHD,
        'AMAZE': rawpy.DemosaicAlgorithm.AMAZE,
        'DCB': rawpy.DemosaicAlgorithm.DCB,
        'DHT': rawpy.DemosaicAlgorithm.DHT,
        'LINEAR': rawpy.DemosaicAlgorithm.LINEAR,
        'LMMSE': rawpy.DemosaicAlgorithm.LMMSE,
        'MODIFIED_AHD': rawpy.DemosaicAlgorithm.MODIFIED_AHD,
        'PPG': rawpy.DemosaicAlgorithm.PPG,
        'VCD': rawpy.DemosaicAlgorithm.VCD,
        'VCD_MODIFIED_AHD': rawpy.DemosaicAlgorithm.VCD_MODIFIED_AHD,
        'VNG': rawpy.DemosaicAlgorithm.VNG
    }
    output_colors = {
        'Adobe': rawpy.ColorSpace.Adobe,
        'ProPhoto': rawpy.ColorSpace.ProPhoto,
        'Wide': rawpy.ColorSpace.Wide,
        'XYZ': rawpy.ColorSpace.XYZ,
        'raw': rawpy.ColorSpace.raw,
        'sRGB': rawpy.ColorSpace.sRGB
    }
    fbdd_noise_reductions = {
        'Full': rawpy.FBDDNoiseReductionMode.Full,
        'Light': rawpy.FBDDNoiseReductionMode.Light,
        'Off': rawpy.FBDDNoiseReductionMode.Off
    }

    demosaic_algorithm = get_value(
        demosaic_algorithms, demosaic_algorithm, 'AHD')
    output_color = get_value(output_colors, output_color, 'sRGB')
    fbdd_noise_reduction = get_value(
        fbdd_noise_reductions, fbdd_noise_reduction, 'Off')

    return {
        'use_camera_wb': use_camera_wb,
        'use_auto_wb': use_auto_wb,
        'bright': bright,
        'median_filter_passes': median_filter_passes,
        'noise_thr': noise_thr,
        'dcb_enhance': dcb_enhance,
        'four_color_rgb': four_color_rgb,
        'half_size': False,
        'demosaic_algorithm': demosaic_algorithm,
        'fbdd_noise_reduction': fbdd_noise_reduction,
        'output_color': output_color,
        'output_bps': output_bps if output_bps == 8 or output_bps == 16 else 8
    }


def resize_image_for_pieces(image, pieces_size=(640, 480)):
    """
    Resizes the image so it can be cutted in similar pieces of size pieces_size
    :param: image is a PIL image
    :return: image resized
    """
    # int(size/piece_size) * piece_size
    new_w = (image.size[0] // pieces_size[0]) * pieces_size[0]
    new_h = (image.size[1] // pieces_size[1]) * pieces_size[1]

    print(new_w, new_h)

    return image.resize((new_w, new_h))


def split_image_pieces(image, pieces_size=(640, 480)):
    """
    Splits an image into similar pieces of size pieces_size
    :return: array of images with size pieces_size
    """
    M = pieces_size[1]
    N = pieces_size[0]

    # assert False
    image = np.asarray(image)

    return [image[x:x+M, y:y+N]
            for x in range(0, M + 1, int(M/2))
            for y in range(0, N + 1, int(N/2))]

# I dont think this works with other split_ratios


def convert_folder(path, pieces_size=(640, 480), split_ratio=4):
    rawpy_params = get_rawpy_params()
    ext = 'jpg'

    files_paths = [join(path, f)
                   for f in listdir(path) if isfile(join(path, f))]
    i = 1

    for image_path in files_paths:
        if 'ARW' in image_path or 'arw' in image_path:
            print('Converting {}'.format(image_path))
            # ------ Convert extension name
            target = image_path.replace('ARW', ext).replace('arw', ext)
            image_name = target.split('/')[-1]

            with rawpy.imread(image_path) as raw:
                # ------ Process image
                rgb = raw.postprocess(**rawpy_params)
                image = Image.fromarray(rgb)

                # ------ Resize image
                image = image.resize((int(pieces_size[0] * split_ratio / 2),
                                      int(pieces_size[1] * split_ratio / 2)))

                # image = resize_image_for_pieces(image,
                #                                 (int(pieces_size[0] * split_ratio / 2),
                #                                 int(pieces_size[1] * split_ratio / 2)))

                # ------ Split image
                pieces = split_image_pieces(image, pieces_size)

                # ------ Save images
                for piece in pieces:
                    piece_target = target.split(
                        image_name)[0] + 'cropped/' + str(i) + '_' + image_name
                    print('Target: {}'.format(piece_target))
                    piece_img = Image.fromarray(piece)
                    piece_img.save(piece_target, quality=100, optimize=True)
                    i += 1


if __name__ == "__main__":
    # Change this path to the one that contains the SeaThru D1 dataset
    path = "/media/data/2022_Noya/datasets/D1_Part1/Raw"
    if not os.path.exists(path + '/cropped/'):
        os.makedirs(path + '/cropped/')
    convert_folder(path)
