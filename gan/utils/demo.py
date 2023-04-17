import torch
from . import load_image_to_eval
from . import save_rgb_histograms, get_rgb_histograms, get_histogram_max_value
from torchvision.utils import save_image
import numpy as np
import os
import matplotlib
import sys

matplotlib.use("Agg")

sys.path.insert(1, "../")

from datasets import get_data


def save_demo(generator, dataset, images_indexes, epoch, params, device):
    # --------- This eval will just affect the generator inside this function
    generator.eval()
    output_path = params["output_image"]["saving_path"]

    for image_index, array_index in zip(
        images_indexes, np.arange(0, len(images_indexes))
    ):
        # --------- Get the images from the dataset
        rgbd_image, _ = get_data(dataset[image_index], device)

        # --------- Conver the image to a batch of size 1
        rgbd_image = torch.unsqueeze(rgbd_image, dim=0)

        # --------- Get the output image
        output_rgb_image = generator(rgbd_image)
        input_rgb_image, _ = generator.split_rgbd(rgbd_image)
        input_rgb_image /= 255

        # --------- Get the histogram of all images
        input_histogram_path = "{0}{1}".format(output_path, "temp_input_histogram.jpg")
        output_histogram_path = "{0}{1}".format(
            output_path, "temp_output_histogram.jpg"
        )

        # --------- Get the histograms from input and output images
        input_hist_r, input_hist_g, input_hist_b = get_rgb_histograms(
            input_rgb_image[0]
        )
        output_hist_r, output_hist_g, output_hist_b = get_rgb_histograms(
            output_rgb_image[0]
        )

        # --------- Get the max value of the histograms
        max_input_value = get_histogram_max_value(
            (input_hist_r, input_hist_g, input_hist_b)
        )
        max_output_value = get_histogram_max_value(
            (output_hist_r, output_hist_g, output_hist_b)
        )
        max_histogram_value = max(max_input_value, max_output_value)

        # --------- Temporaly save the histograms to disk
        save_rgb_histograms(
            None,
            input_histogram_path,
            "Input image histogram",
            (input_hist_r, input_hist_g, input_hist_b),
            max_histogram_value,
        )
        save_rgb_histograms(
            None,
            output_histogram_path,
            "Output image histogram",
            (output_hist_r, output_hist_g, output_hist_b),
            max_histogram_value,
        )

        # --------- Load the histograms from disk
        input_histogram = load_image_to_eval(input_histogram_path, device)
        output_histogram = load_image_to_eval(output_histogram_path, device)

        # --------- Delete the histograms from disk
        os.remove(input_histogram_path)
        os.remove(output_histogram_path)

        # --------- Creating the grid
        demo_grid = torch.cat(
            (
                input_rgb_image,
                output_rgb_image,
                input_histogram,
                output_histogram,
            ),
            0,
        )

        # --------- Saving the grid
        save_image(
            demo_grid,
            "{0}{1}-{2}.jpg".format(output_path, epoch, array_index),
            nrow=2,
        )
