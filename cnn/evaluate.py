import json
import os
import torch
import copy
import csv
import numpy as np
from utils import (
    create_saving_paths,
    calculate_uw_metrics,
    save_rgb_histograms,
    load_image_to_eval,
)
from models import UNet
from datasets import EvaluateDataLoaderCreator
from torchvision.utils import save_image

DATASET_TO_USE = "real-underwater-c"

# ---------- Opening parameters
with open(os.path.dirname(__file__) + "parameters.json") as path_file:
    params = json.load(path_file)

# ---------- Creating paths for saved images
create_saving_paths(params, True)
output_path = params["output_evaluation"][DATASET_TO_USE]

# ---------- Checking for CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- Dataset
params["data_loader"]["shuffle"] = False
params["data_loader"]["batch_size"] = 1
data_loader = EvaluateDataLoaderCreator(params, params["datasets"][DATASET_TO_USE])
all_dataset_loader, _ = data_loader.get_loaders()

# ---------- Models
unet = UNet(params).to(device)
unet.eval()

# ---------- Loading weights
loaded_weights = copy.deepcopy(
    torch.load(params["unet"]["saving_path"] + "unet.pt", device)
)
unet.load_state_dict(loaded_weights)

# ---------- Counter used for image tracking
counter = 0

# ---------- CSV results variables
CSV_PATH = output_path + "./results.csv"
if os.path.exists(CSV_PATH):
    os.remove(CSV_PATH)

csv_file = open(CSV_PATH, "a")
csv_writer = csv.writer(csv_file)
csv_headers = [
    "name",
    "uicm_before",
    "uism_before",
    "uiconm_before",
    "uiqm_before",
    "uicm_after",
    "uism_after",
    "uiconm_after",
    "uiqm_after",
]
csv_writer.writerow(csv_headers)

# ---------- Evaluation store variables
uicm_before = np.array([])
uism_before = np.array([])
uiconm_before = np.array([])
uiqm_before = np.array([])

uicm_after = np.array([])
uism_after = np.array([])
uiconm_after = np.array([])
uiqm_after = np.array([])

for i, data in enumerate(all_dataset_loader, 0):
    # ------ Get the data from the data_loader
    raw_images = data
    raw_images = raw_images.to(device)

    with torch.no_grad():
        recolored_images = unet(raw_images)
        recolored_images = torch.clamp(recolored_images, min=0.0, max=1.0)

    for raw, recolored in zip(raw_images, recolored_images):
        print(counter)
        plot_output_path = params["output_image"]["real"]["saving_path"]

        # --------- Get the histogram of both images
        real_histogram_path = "{0}temp_real_histogram.jpg".format(plot_output_path)
        recolored_histogram_path = "{0}temp_recolored_histogram.jpg".format(
            plot_output_path
        )

        # --------- Temporaly save the histogram to disk
        save_rgb_histograms(
            raw,
            real_histogram_path,
            "Raw image histogram",
        )
        save_rgb_histograms(
            recolored,
            recolored_histogram_path,
            "Recolored image histogram",
        )

        # --------- Load the histograms from disk
        real_histogram = load_image_to_eval(
            real_histogram_path,
            device,
            size=(raw.shape[3 - 1], raw.shape[2 - 1]),
        )
        recolored_histogram = load_image_to_eval(
            recolored_histogram_path,
            device,
            size=(raw.shape[3 - 1], raw.shape[2 - 1]),
        )
        # --------- Delete the histograms from disk
        os.remove(real_histogram_path)
        os.remove(recolored_histogram_path)

        # --------- Creating the grid
        images_grid = torch.cat((raw, recolored), 2)
        histograms_grid = torch.cat((real_histogram[0], recolored_histogram[0]), 2)
        real_grid = torch.cat((images_grid, histograms_grid), 1)

        # --------- Saving the grid
        save_image(
            real_grid,
            "{0}{1}.jpg".format(output_path, counter),
            nrow=2,
        )

        np_raw, np_recolored = (
            np.transpose(raw.cpu().detach().numpy(), (1, 2, 0)) * 255,
            np.transpose(recolored.cpu().detach().numpy(), (1, 2, 0)) * 255,
        )
        np_raw, np_recolored = (np_raw.astype(np.uint8), np_recolored.astype(np.uint8))

        uicm_b, uism_b, uiconm_b, uiqm_b = calculate_uw_metrics(np_raw)

        uicm_before = np.append(uicm_before, uicm_b)
        uism_before = np.append(uism_before, uism_b)
        uiconm_before = np.append(uiconm_before, uiconm_b)
        uiqm_before = np.append(uiqm_before, uiqm_b)

        print(
            "\nRaw metrics: \nUICM: {0:.6f}\nUISM: {1:.6f}\nUICONM: {2:.6f}\nUIQM: {3:.6f}".format(
                uicm_b, uism_b, uiconm_b, uiqm_b
            )
        )

        uicm_a, uism_a, uiconm_a, uiqm_a = calculate_uw_metrics(np_recolored)

        print(
            "\nRecolored metrics: \nUICM: {0:.6f}\nUISM: {1:.6f}\nUICONM: {2:.6f}\nUIQM: {3:.6f} \n\n\n".format(
                uicm_a, uism_a, uiconm_a, uiqm_a
            )
        )

        uicm_after = np.append(uicm_after, uicm_a)
        uism_after = np.append(uism_after, uism_a)
        uiconm_after = np.append(uiconm_after, uiconm_a)
        uiqm_after = np.append(uiqm_after, uiqm_a)

        csv_writer.writerow(
            [
                counter,
                format(uicm_b, ".6f"),
                format(uism_b, ".6f"),
                format(uiconm_b, ".6f"),
                format(uiqm_b, ".6f"),
                format(uicm_a, ".6f"),
                format(uism_a, ".6f"),
                format(uiconm_a, ".6f"),
                format(uiqm_a, ".6f"),
            ]
        )

        counter += 1

average_uicm_before = np.average(uicm_before)
average_uism_before = np.average(uism_before)
average_uiconm_before = np.average(uiconm_before)
average_uiqm_before = np.average(uiqm_before)

average_uicm_after = np.average(uicm_after)
average_uism_after = np.average(uism_after)
average_uiconm_after = np.average(uiconm_after)
average_uiqm_after = np.average(uiqm_after)

print(
    "\nBEFORE STATS:\nAverage UICM: {0:.6f}\nAverage UISM: {1:.6f}\nAverage UICONM: {2:.6f}\nAverage UIQM: {3:.6f} \n\n\n".format(
        average_uicm_before,
        average_uism_before,
        average_uiconm_before,
        average_uiqm_before,
    )
)


print(
    "\nAFTER STATS:\nAverage UICM: {0:.6f}\nAverage UISM: {1:.6f}\nAverage UICONM: {2:.6f}\nAverage UIQM: {3:.6f} \n\n\n".format(
        average_uicm_after, average_uism_after, average_uiconm_after, average_uiqm_after
    )
)

csv_writer.writerow(
    [
        "Average",
        format(average_uicm_before, ".6f"),
        format(average_uism_before, ".6f"),
        format(average_uiconm_before, ".6f"),
        format(average_uiqm_before, ".6f"),
        format(average_uicm_after, ".6f"),
        format(average_uism_after, ".6f"),
        format(average_uiconm_after, ".6f"),
        format(average_uiqm_after, ".6f"),
    ]
)

csv_file.close()
