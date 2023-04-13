import torch
from . import load_image_to_eval
from . import save_rgb_histograms
from torchvision.utils import save_image
import numpy as np
import os
import matplotlib

matplotlib.use("Agg")


def save_demo(generator, dataset, images_indexes, epoch, params, device):
    # --------- This eval will just affect the generator inside this function
    generator.eval()
    output_path = params["output_image"]["saving_path"]

    # --------- Get the images from the dataset
    rgbd_images = dataset[images_indexes]

    for rgbd_image, index in zip(rgbd_images, np.arange(0, len(rgbd_images))):
        output_rgb_image = generator(rgbd_image)
        input_rgb_image, _ = generator.split_rgbd(rgbd_image)

        # --------- Get the histogram of all images
        input_histogram_path = "{0}{1}".format(output_path, "temp_input_histogram.jpg")
        output_histogram_path = "{0}{1}".format(
            output_path, "temp_output_histogram.jpg"
        )

        # --------- Temporaly save the histograms to disk
        save_rgb_histograms(
            input_rgb_image[0],
            input_histogram_path,
            "Input image histogram",
        )
        save_rgb_histograms(
            output_rgb_image[0],
            output_histogram_path,
            "Output image histogram",
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
            "{0}{1}-{2}.jpg".format(output_path, epoch, index),
            nrow=2,
        )
