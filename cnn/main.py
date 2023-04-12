import json
import torch
import os
from torchvision.utils import save_image
from datasets import DataLoaderCreator, get_data
from models import UNet
from utils import (
    DataHandler,
    create_saving_paths,
    save_real_demo,
    save_synthetic_demo,
)

DEMO_REAL_IMAGES = ["D1P1_T_S03077_2.jpg", "D1P1_T_S03050_4.jpg", "D2P2_T_S03697_1.jpg"]
DEMO_SYNTHETIC_IMAGES = ["789.png", "995.png", "807.png"]

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
unet_handler = DataHandler(True, None)


try:
    # ---------- Training epochs
    for epoch in range(params["epochs"]):
        # ------------------- UNET
        for i, data in enumerate(training_loader, 0):
            # ------ Get the data from the data_loader
            image, gt = get_data(data, device)

            # ------ Train
            unet.train()
            y_hat, loss = unet.fit(image, gt)

            # ------ Save train loss
            unet_handler.append_train_loss(loss)

        if epoch == 0 or epoch % params["epochs_checkpoint"] == 0:
            # --------- Saving some demo images
            save_real_demo(unet, epoch, DEMO_REAL_IMAGES, params, device)
            save_synthetic_demo(unet, epoch, DEMO_SYNTHETIC_IMAGES, params, device)

        # ----- Handle epoch ending
        unet_handler.epoch_end(epoch, unet.lr)

except KeyboardInterrupt:
    pass

unet_handler.save_data(params["output_stats"]["saving_path"])
