import json
import os
import torch
import copy
import numpy as np
from utils import create_saving_paths, calculate_uw_metrics
from models import UNet
from datasets import EvaluateDataLoaderCreator

DATASET_TO_USE = "real-underwater-a"

# ---------- Opening parameters
with open(os.path.dirname(__file__) + "parameters.json") as path_file:
    params = json.load(path_file)

# ---------- Creating paths for saved images
create_saving_paths(params, True)

# ---------- Checking for CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- Dataset
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

global_uicm = np.array([])
global_uism = np.array([])
global_uiconm = np.array([])
global_uiqm = np.array([])

for i, data in enumerate(all_dataset_loader, 0):
    # ------ Get the data from the data_loader
    raw_images = data
    raw_images = raw_images.to(device)

    with torch.no_grad():
        recolored_images = unet(raw_images)

    for raw, recolored in zip(raw_images, recolored_images):
        print(counter)
        uicm, uism, uiconm, uiqm = calculate_uw_metrics(raw)

        print(
            "\nRaw metrics: \nUICM: {0:.2f}\nUISM: {1:.2f}\nUICONM: {2:.2f}\nUIQM: {3:.2f}".format(
                uicm, uism, uiconm, uiqm
            )
        )

        uicm, uism, uiconm, uiqm = calculate_uw_metrics(recolored)

        print(
            "\nRecolored metrics: \nUICM: {0:.2f}\nUISM: {1:.2f}\nUICONM: {2:.2f}\nUIQM: {3:.2f} \n\n\n".format(
                uicm, uism, uiconm, uiqm
            )
        )

        global_uicm = np.append(global_uicm)
        global_uism = np.append(global_uism)
        global_uiconm = np.append(global_uiconm)
        global_uiqm = np.append(global_uiqm)

        counter += 1

average_uicm = np.average(global_uicm)
average_uism = np.average(global_uism)
average_uiconm = np.average(global_uiconm)
average_uiqm = np.average(global_uiqm)


print(
    "\nAverage UICM: {0:.2f}\nAverage UISM: {1:.2f}\nAverage UICONM: {2:.2f}\nAverage UIQM: {3:.2f} \n\n\n".format(
        average_uicm, average_uism, average_uiconm, average_uiqm
    )
)
