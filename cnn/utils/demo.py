import torch
from . import load_image_to_eval
from . import save_rgb_histograms
from torchvision.utils import save_image
import numpy as np
import os


def save_real_demo(unet, epoch, images_path, params, device):
    unet.eval()
    # --------- Saving a recolored real image
    with torch.no_grad():
        for image, index in zip(images_path, np.arange(0, len(images_path))):
            real_image = load_image_to_eval(
                params["datasets"]["underwater"] + image, device
            )

            # --------- Recoloring the image
            recolored_image = unet(real_image)

            # --------- Get the histogram of both images
            real_histogram_path = (
                params["output_image"]["real"]["saving_path"]
                + "temp_real_histogram.jpg"
            )

            recolored_histogram_path = (
                params["output_image"]["real"]["saving_path"]
                + "temp_recolored_histogram.jpg"
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
            real_histogram = load_image_to_eval(real_histogram_path, device)
            recolored_histogram = load_image_to_eval(recolored_histogram_path, device)

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
                params["output_image"]["real"]["saving_path"]
                + str(epoch)
                + "-"
                + str(index)
                + ".jpg",
                nrow=2,
            )


def save_synthetic_demo(unet, epoch, images_path, params, device):
    unet.eval()
    # --------- Saving a recolored real image
    with torch.no_grad():
        for image, index in zip(images_path, np.arange(0, len(images_path))):
            input_image = load_image_to_eval(
                params["datasets"]["synthetic"] + "images/" + image, device
            )
            gt_image = load_image_to_eval(
                params["datasets"]["synthetic"] + "gt/" + image, device
            )

            # --------- Recoloring the image
            output_image = unet(input_image)

            # --------- Get the histogram of all images
            input_histogram_path = (
                params["output_image"]["synthetic"]["saving_path"]
                + "temp_input_histogram.jpg"
            )
            gt_histogram_path = (
                params["output_image"]["synthetic"]["saving_path"]
                + "temp_gt_histogram.jpg"
            )
            output_histogram_path = (
                params["output_image"]["synthetic"]["saving_path"]
                + "temp_output_histogram.jpg"
            )

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
            input_histogram = load_image_to_eval(input_histogram_path, device)
            gt_histogram = load_image_to_eval(gt_histogram_path, device)
            output_histogram = load_image_to_eval(output_histogram_path, device)

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
                params["output_image"]["synthetic"]["saving_path"]
                + str(epoch)
                + "-"
                + str(index)
                + ".jpg",
                nrow=3,
            )
