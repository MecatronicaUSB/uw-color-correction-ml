import json
import torch
import os
import numpy as np
from torchvision.utils import save_image
from matplotlib import pyplot as plt
from datasets import UNETDataset, DataLoaderCreator, get_data
from models import UNet
from utils import np_utils, data_handler
from torchvision.utils import save_image

# ---------- Opening parameters
with open(os.path.dirname(__file__) + "parameters.json") as path_file:
    params = json.load(path_file)

# ---------- Checking for CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- Dataset
dataloader_creator = DataLoaderCreator(params)
training_loader, validation_loader = dataloader_creator.get_loaders()

# ---------- Models
unet = UNet(params["unet"]).to(device)

loss_function = torch.nn.BCELoss()
handler = data_handler.DataHandler(True, None)

# ---------- Training epochs
for epoch in range(params["epochs"]):
    # ------------------- UNET
    for i, data in enumerate(training_loader, 0):
        # ------ Get the data from the data_loader
        image, gt = get_data(data, device)

        # ------ Train
        unet.train()
        y_hat, loss = unet.fit(image, gt)

        handler.append_train_loss(loss)

    if epoch % 10 == 0:
        save_image(y_hat, "images/{}.png".format(epoch), nrow=1)

    handler.epoch_end(epoch, unet.lr)

# handler.plot(False, False)
