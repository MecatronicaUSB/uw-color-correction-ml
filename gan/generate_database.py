import json
import torch
import os
import copy

from models import Generator
from datasets import DataLoaderCreator, get_data
from torchvision.utils import save_image

# ---------- Opening parameters
with open(os.path.dirname(__file__) + "parameters.json") as path_file:
    params = json.load(path_file)

# ---------- Checking for CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- Dataset
params["train_percentage"] = 1  # We set this to 1 for this script
dataloader_creator = DataLoaderCreator(params)
all_dataset_loader, _ = dataloader_creator.get_loaders()

# ---------- Model
generator = Generator(params["generator"]).to(device)

# ---------- Loading weights
loaded_weights = copy.deepcopy(torch.load(
    params["generator"]["saving_path"], device))
generator.load_state_dict(loaded_weights)

# ---------- Evaluation mode
generator.eval()

# ---------- Counter used for image naming
counter = 0

for i, data in enumerate(all_dataset_loader, 0):
    # ------ Get the data from the data_loader
    in_air, _ = get_data(data, device)

    with torch.no_grad():
        rgb_images, _ = generator.split_rgbd(in_air)
        synthetic_images = generator(in_air)

    for rgb, synthetic in zip(rgb_images, synthetic_images):
        print(counter)
        save_image(rgb / 255, params["datasets"]["synthetic"] +
                   "/gt/" + str(counter) + ".png", nrow=1)
        save_image(synthetic, params["datasets"]["synthetic"] +
                   "/images/" + str(counter) + ".png", nrow=1)

        counter += 1
