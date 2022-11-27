import json
import torch
import os
import numpy as np
from torchvision.utils import save_image
from matplotlib import pyplot as plt

from models import Discriminator, Generator
from datasets import UWGANDataset, DataLoaderCreator
from utils import np_utils, data_handler
from training import train, validate_d, get_data

# ---------- Opening parameters
with open(os.path.dirname(__file__) + "parameters.json") as path_file:
    params = json.load(path_file)

# ---------- Checking for CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- Dataset
dataloader_creator = DataLoaderCreator(params)
training_loader, validation_loader = dataloader_creator.get_loaders()

# ---------- Models
generator = Generator().to(device)
discriminator = Discriminator().to(device)

loss_function = torch.nn.BCELoss()
handler = data_handler.DataHandler(True, True)

# ---------- Training epochs
for epoch in range(params["epochs"]):
    # ------------------- Training the GAN --------------------- #
    for i, data in enumerate(training_loader, 0):
        # ------ Get the data from the data_loader
        in_air, underwater = get_data(data, device)

        fake_underwater, g_loss, d_loss = train(
            generator, discriminator, in_air, underwater, loss_function, device
        )
        handler.append_train_loss(d_loss.item())

    # ------------------- Validatin the GAN--------------------- #
    for i, data in enumerate(validation_loader, 0):
        # ------ Get the data from the data_loader
        in_air, underwater = get_data(data, device)

        d_loss = validate_d(
            generator, discriminator, in_air, underwater, loss_function, device
        )
        handler.append_valid_loss(d_loss.item())

    handler.epoch_end(epoch, 0)

handler.plot(True, False)
