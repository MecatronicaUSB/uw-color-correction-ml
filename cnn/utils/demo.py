import torch
from . import load_image_to_eval
from . import save_rgb_histograms
from torchvision.utils import save_image
import numpy as np
import os


def save_real_demo(unet, epoch, images_path, params, device):
    unet.eval()
    dataset_path = params["datasets"]["underwater"]
    output_path = params["output_image"]["real"]["saving_path"]

    # --------- Saving a recolored real image
    with torch.no_grad():
        for image, index in zip(images_path, np.arange(0, len(images_path))):
            real_image = load_image_to_eval(dataset_path + image, device)

            # --------- Recoloring the image
            recolored_image = unet(real_image)

            # --------- Get the histogram of both images
            real_histogram_path = "{0}temp_real_histogram.jpg".format(output_path)

            recolored_histogram_path = "{0}temp_recolored_histogram.jpg".format(
                output_path
            )

            # --------- Temporaly save the histogram to disk
            save_rgb_histograms(
                real_image[0],
                real_histogram_path,
                "Real image histogram",
            )
            save_rgb_histograms(
                recolored_image[0],
                recolored_histogram_path,
                "Recolored image histogram",
            )

            # --------- Load the histograms from disk
            real_histogram = load_image_to_eval(
                real_histogram_path,
                device,
                size=(real_image.shape[3], real_image.shape[2]),
            )
            recolored_histogram = load_image_to_eval(
                recolored_histogram_path,
                device,
                size=(real_image.shape[3], real_image.shape[2]),
            )

            # --------- Delete the histograms from disk
            os.remove(real_histogram_path)
            os.remove(recolored_histogram_path)

            # --------- Creating the grid
            real_grid = torch.cat(
                (real_image, recolored_image, real_histogram, recolored_histogram), 0
            )

            # --------- Saving the grid
            save_image(
                real_grid,
                "{0}{1}-{2}.jpg".format(output_path, epoch, index),
                nrow=2,
            )


def save_synthetic_demo(unet, epoch, images_path, params, device):
    unet.eval()
    dataset_path = params["datasets"]["synthetic"]
    output_path = params["output_image"]["synthetic"]["saving_path"]

    # --------- Saving a recolored real image
    with torch.no_grad():
        for image, index in zip(images_path, np.arange(0, len(images_path))):
            input_image = load_image_to_eval(
                "{0}images/{1}".format(dataset_path, image), device
            )
            gt_image = load_image_to_eval(
                "{0}gt/{1}".format(dataset_path, image), device
            )

            # --------- Recoloring the image
            output_image = unet(input_image)

            # --------- Get the histogram of all images
            input_histogram_path = "{0}temp_input_histogram.jpg".format(output_path)
            gt_histogram_path = "{0}temp_gt_histogram.jpg".format(output_path)
            output_histogram_path = "{0}temp_output_histogram.jpg".format(output_path)

            # --------- Temporaly save the histograms to disk
            save_rgb_histograms(
                input_image[0],
                input_histogram_path,
                "Input image histogram",
            )
            save_rgb_histograms(
                gt_image[0],
                gt_histogram_path,
                "GT image histogram",
            )
            save_rgb_histograms(
                output_image[0],
                output_histogram_path,
                "Output image histogram",
            )

            # --------- Load the histograms from disk
            input_histogram = load_image_to_eval(
                input_histogram_path,
                device,
                size=(input_image.shape[3], input_image.shape[2]),
            )
            gt_histogram = load_image_to_eval(
                gt_histogram_path,
                device,
                size=(input_image.shape[3], input_image.shape[2]),
            )
            output_histogram = load_image_to_eval(
                output_histogram_path,
                device,
                size=(input_image.shape[3], input_image.shape[2]),
            )

            # --------- Delete the histograms from disk
            os.remove(input_histogram_path)
            os.remove(gt_histogram_path)
            os.remove(output_histogram_path)

            # --------- Creating the grid
            real_grid = torch.cat(
                (
                    input_image,
                    output_image,
                    gt_image,
                    input_histogram,
                    output_histogram,
                    gt_histogram,
                ),
                0,
            )

            # --------- Saving the grid
            save_image(
                real_grid,
                "{0}{1}-{2}.jpg".format(
                    output_path,
                    epoch,
                    index,
                ),
                nrow=3,
            )
