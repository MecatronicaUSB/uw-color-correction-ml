import json
import torch
import os
import numpy as np
import copy
from torchvision.utils import save_image
from matplotlib import pyplot as plt
from datasets import UNETDataset, DataLoaderCreator, get_data
from models import UNet
from utils import np_utils, data_handler
from parsers import usingPILandShrink
from torchvision.utils import save_image

# ---------- Opening parameters
with open(os.path.dirname(__file__) + "parameters.json") as path_file:
    params = json.load(path_file)

# ---------- Checking for CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- Setting epoch_checkpoints
# if "epoch_checkpoints" not in params:
#     # Default value - as in original release
#     params["epoch_checkpoints"] = 10
# else:
#     # Check that the value is a positive integer
#     if not isinstance(params["epoch_checkpoints"], int) or params["epoch_checkpoints"] < 1:
#         raise ValueError("epoch_checkpoints must be a positive integer")
#     params["epoch_checkpoints"] = int(params["epoch_checkpoints"])

# ---------- Dataset
dataloader_creator = DataLoaderCreator(params)
training_loader, validation_loader = dataloader_creator.get_loaders()

# ---------- Models
unet = UNet(params["unet"]).to(device)

loss_function = torch.nn.BCELoss()
handler = data_handler.DataHandler(True, None)

# ---------- Creating paths for demo images
if not os.path.exists('./images/'):
    os.makedirs('./images/')

if not os.path.exists('./images/synthetic'):
    os.makedirs('./images/synthetic')

if not os.path.exists('./images/real'):
    os.makedirs('./images/real')

# ---------- Training epochs
for epoch in range(params["epochs"]):
    image = None
    gt = None
    y_hat = None

    # ------------------- UNET
    for i, data in enumerate(training_loader, 0):
        # ------ Get the data from the data_loader
        image, gt = get_data(data, device)

        # ------ Train
        unet.train()
        y_hat, loss = unet.fit(image, gt)

        handler.append_train_loss(loss)

    # --------- Saving some demo images
    if epoch != 0 and epoch % params["epochs_checkpoint"] == 0:
        print('\nSaving images...')
        grid = torch.cat((image, gt, y_hat), 0)
        save_image(grid, "./images/synthetic/{}.png".format(epoch),
                   nrow=image.shape[0])

        # --------- Saving a real image recolored
        unet.eval()
        with torch.no_grad():
            image = np.asarray(usingPILandShrink(
                params["datasets"]["underwater"] + "/D1P1_T_S03077_2.jpg", None))
            image = np_utils.transpose_hwc_to_chw(image)
            image = np_utils.add_channel_first(image)
            image = torch.from_numpy(copy.deepcopy(
                image)).float().to(device) / 255
            eval_image = unet(image)
            real_grid = torch.cat((image, eval_image), 0)
            save_image(
                real_grid, "./images/real/{}.jpg".format(epoch), nrow=2)

    handler.epoch_end(epoch, unet.lr)

# handler.plot(False, False)
