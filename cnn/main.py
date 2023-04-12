import json
import torch
import os
from torchvision.utils import save_image
from datasets import DataLoaderCreator, get_data
from models import UNet
from utils import (
    DataHandler,
    create_saving_paths,
    load_image_to_eval,
    save_rgb_histograms,
)

DEMO_REAL_IMAGE = "D1P1_T_S03077_2.jpg"

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

            unet_handler.append_train_loss(loss)

        # --------- Saving some demo images
        if epoch == 0 or epoch % params["epochs_checkpoint"] == 0:
            # unet.save_weights(epoch)
            unet.eval()

            # --------- Testing saving histogram
            save_rgb_histograms(
                image[0],
                params["output_stats"]["saving_path"] + "input-" + str(epoch) + ".png",
                "Input image Histogram",
            )
            save_rgb_histograms(
                gt[0],
                params["output_stats"]["saving_path"] + "gt-" + str(epoch) + ".png",
                "GT image Histogram",
            )
            save_rgb_histograms(
                y_hat[0],
                params["output_stats"]["saving_path"] + "output-" + str(epoch) + ".png",
                "Output image Histogram",
            )

            # --------- Saving a recolored synthetic image
            grid = torch.cat((image, gt, y_hat), 0)
            save_image(
                grid,
                params["output_image"]["synthetic"]["saving_path"]
                + str(epoch)
                + ".png",
                nrow=image.shape[0],
            )

            # --------- Saving a recolored real image
            with torch.no_grad():
                real_image = load_image_to_eval(
                    params["datasets"]["underwater"] + DEMO_REAL_IMAGE, device
                )

                # --------- Recoloring the image
                recolored_image = unet(real_image)

                # --------- Creating the grid
                real_grid = torch.cat((real_image, recolored_image), 0)

                # --------- Saving the grid
                save_image(
                    real_grid,
                    params["output_image"]["real"]["saving_path"] + str(epoch) + ".png",
                    nrow=2,
                )

        unet_handler.epoch_end(epoch, unet.lr)

except KeyboardInterrupt:
    pass

unet_handler.save_data(params["output_stats"]["saving_path"])
