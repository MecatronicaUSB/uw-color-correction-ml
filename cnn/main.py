import json
import torch
import math
import os
import warnings
from datasets import DataLoaderCreator, get_data
from models import UNet
from utils import (
    DataHandler,
    create_saving_paths,
    save_real_demo,
    save_synthetic_demo,
)
import matplotlib

matplotlib.use("Agg")


DEMO_REAL_IMAGES = ["D1P1_T_S03077_2.jpg", "D1P1_T_S03050_4.jpg", "D2P2_T_S03697_1.jpg"]
DEMO_SYNTHETIC_IMAGES = ["789.png", "995.png", "807.png"]

# ---------- Ignore PyTorch warning
warnings.simplefilter("ignore", UserWarning)

# ---------- Opening parameters
with open(os.path.dirname(__file__) + "parameters.json") as path_file:
    params = json.load(path_file)

# ---------- Creating paths for saved images and weights
create_saving_paths(params)

# ---------- Checking for CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- Dataset
data_loader = DataLoaderCreator(params)
training_loader, validation_loader = data_loader.get_loaders()

# ---------- Models
unet = UNet(params["unet"]).to(device)

# ---------- Init data handler
unet_handler = DataHandler(validation=True)

try:
    # ---------- Training epochs
    for epoch in range(params["epochs"]):
        print("\n\n-------------- Epoch: {0} --------------".format(epoch))

        # ------------------- UNET Training ------------------- #
        for i, data in enumerate(training_loader, 0):
            # ------ Get the data from the data_loader
            image, gt = get_data(data, device)

            # ------ Train
            unet.train()
            y_hat, loss = unet.fit(image, gt)

            # ------ Save train loss
            unet_handler.append_train_loss(loss)

        # ------------------- UNET Validation ------------------- #
        for i, data in enumerate(validation_loader, 0):
            # ------ Get the data from the data_loader
            image, gt = get_data(data, device)

            # ------ Evaulate
            unet.eval()

            with torch.no_grad():
                loss = unet.evaluate(image, gt)

                # ------ Save train loss
                unet_handler.append_valid_loss(loss)

        # ------------------- Save demo images ------------------- #
        if epoch == 0 or epoch % params["epochs_checkpoint"] == 0:
            # --------- Saving real and synthethic demo images
            save_real_demo(unet, epoch, DEMO_REAL_IMAGES, params, device)
            save_synthetic_demo(unet, epoch, DEMO_SYNTHETIC_IMAGES, params, device)

        # ----- Handle epoch ending
        is_best_valid_loss = unet_handler.epoch_end(epoch, unet.lr)

        if is_best_valid_loss:
            print("Saving weights to {0}".format(unet.saving_path))
            unet.save_weights()

except KeyboardInterrupt:
    pass

print("Saving loss charts to {0}".format(params["output_stats"]["saving_path"]))
unet_handler.save_data(params["output_stats"]["saving_path"])
unet_handler.save_data_from_epoch(
    math.ceil(epoch * 0.6), params["output_stats"]["saving_path"]
)
