import json
import torch
import os
import copy

from models import Generator
from datasets import (
    NYUDataLoaderCreator,
)
from torchvision.utils import save_image

# ---------- Opening parameters
with open(os.path.dirname(__file__) + "parameters.json") as path_file:
    params = json.load(path_file)

# ---------- Creating output paths
output_dataset_path = params["datasets"]["synthetic"]

if not os.path.exists(output_dataset_path):
    os.makedirs(output_dataset_path)

if not os.path.exists(output_dataset_path + "/images"):
    os.makedirs(output_dataset_path + "/images")

if not os.path.exists(output_dataset_path + "/gt"):
    os.makedirs(output_dataset_path + "/gt")

# ---------- Checking for CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- Dataset
params["train_percentage"] = 1  # We set this to 1 for this script
params["nyu_data_loader"]["shuffle"] = False  # We set this to False for this script
params["nyu_data_loader"][
    "augmentation"
] = False  # We set this to False for this script
params["nyu_data_loader"]["force_crop"] = True  # We set this to True for this script
dataloader_creator = NYUDataLoaderCreator(params, device)
all_dataset_loader, _ = dataloader_creator.get_loaders()

# ---------- Model
generator = Generator(params["generator"]).to(device)

# ---------- Loading weights
loaded_weights = copy.deepcopy(
    torch.load(params["generator"]["saving_path"] + "/generator-28.pt", device)
)
generator.load_state_dict(loaded_weights)

# ---------- Evaluation mode
generator.eval()

# ---------- Counter used for image naming
counter = 0

for i, data in enumerate(all_dataset_loader, 0):
    # ------ Get the data from the data_loader
    in_air, depth = data

    with torch.no_grad():
        synthetic_images = generator(in_air, depth)

    for rgb, synthetic in zip(in_air, synthetic_images):
        print(counter)
        save_image(
            rgb,
            "{0}gt/{1}.jpg".format(output_dataset_path, counter),
            nrow=1,
        )
        save_image(
            synthetic,
            "{0}/images/{1}.jpg".format(output_dataset_path, counter),
            nrow=1,
        )

        counter += 1
