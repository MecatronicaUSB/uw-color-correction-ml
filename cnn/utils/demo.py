import torch
from image_utils import load_image_to_eval
from histogram import save_rgb_histograms
from torchvision.utils import save_image
import os


def save_real_demo(unet, epoch, images_path, params, device):
    unet.eval()
    # --------- Saving a recolored real image
    with torch.no_grad():
        for image in images_path:
            real_image = load_image_to_eval(
                params["datasets"]["underwater"] + image, device
            )

            # --------- Recoloring the image
            recolored_image = unet(real_image)

            # --------- Get the histogram of both images
            real_histogram_path = (
                params["output_image"]["real"]["saving_path"]
                + "real_histogram_temp.png"
            )

            recolored_histogram_path = (
                params["output_image"]["real"]["saving_path"]
                + "recolored_histogram_temp.png"
            )

            # --------- Temporaly save the histogram to disk
            save_rgb_histograms(
                real_image,
                real_histogram_path,
                "Real image histogram",
            )
            save_rgb_histograms(
                recolored_image,
                recolored_histogram_path,
                "Recolored image histogram",
            )

            # --------- Load the histograms from disk
            real_histogram = load_image_to_eval(real_histogram_path, device)
            recolored_histogram = load_image_to_eval(recolored_histogram_path, device)

            # --------- Creating the grid
            real_grid = torch.cat(
                (real_image, real_histogram, recolored_image, recolored_histogram), 0
            )

            # --------- Saving the grid
            save_image(
                real_grid,
                params["output_image"]["real"]["saving_path"] + str(epoch) + ".png",
                nrow=2,
            )

            # --------- Delete the histograms from disk
            os.remove(real_histogram_path)
            os.remove(recolored_histogram_path)


def save_synthetic_demo(unet, epoch, images_path, params, device):
    unet.eval()
    # --------- Saving a recolored real image
    with torch.no_grad():
        for image in images_path:
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
                + "input_histogram_temp.png"
            )
            gt_histogram_path = (
                params["output_image"]["synthetic"]["saving_path"]
                + "gt_histogram_temp.png"
            )
            output_histogram_path = (
                params["output_image"]["synthetic"]["saving_path"]
                + "output_histogram_temp.png"
            )

            # --------- Temporaly save the histograms to disk
            save_rgb_histograms(
                input_image,
                input_histogram_path,
                "Input image histogram",
            )
            save_rgb_histograms(
                gt_image,
                gt_histogram_path,
                "GT image histogram",
            )
            save_rgb_histograms(
                output_image,
                output_histogram_path,
                "Output image histogram",
            )

            # --------- Load the histograms from disk
            input_histogram = load_image_to_eval(input_histogram_path, device)
            gt_histogram = load_image_to_eval(gt_histogram_path, device)
            output_histogram = load_image_to_eval(output_histogram_path, device)

            # --------- Creating the grid
            real_grid = torch.cat(
                (
                    input_image,
                    input_histogram,
                    output_image,
                    output_histogram,
                    gt_image,
                    gt_histogram,
                ),
                0,
            )

            # --------- Saving the grid
            save_image(
                real_grid,
                params["output_image"]["synthetic"]["saving_path"]
                + str(epoch)
                + ".png",
                nrow=3,
            )

            # --------- Delete the histograms from disk
            os.remove(input_histogram_path)
            os.remove(gt_histogram_path)
            os.remove(output_histogram_path)
