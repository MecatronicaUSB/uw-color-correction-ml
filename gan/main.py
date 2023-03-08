import json
import torch
import os
from torchvision.utils import save_image

from models import Discriminator, Generator
from datasets import DataLoaderCreator, get_data
from utils import data_handler, save_grid

# ---------- Opening parameters
with open(os.path.dirname(__file__) + "parameters.json") as path_file:
    params = json.load(path_file)

# ---------- Checking for CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- Dataset
dataloader_creator = DataLoaderCreator(params)
training_loader, validation_loader = dataloader_creator.get_loaders()

# ---------- Models
generator = Generator(params["generator"]).to(device)
discriminator = Discriminator(params["discriminator"]).to(device)

# ---------- Init data handlers
generator_data_handler = data_handler.DataHandler('Generator')
discriminator_data_handler = data_handler.DataHandler('Discriminator')

# ---------- Initial training states
training_generator = False
training_discriminator = True

# ---------- Training epochs
for epoch in range(params["epochs"]):
    # ------------------- Training the GAN --------------------- #
    for i, data in enumerate(training_loader, 0):
        # ------ Train mode
        generator.train()
        discriminator.train()

        # ------ Get the data from the data_loader
        in_air, underwater = get_data(data, device)

        # ------ Fit the models
        g_loss, fake_underwater = generator.fit(
            discriminator, in_air, training_generator)
        d_loss = discriminator.fit(
            underwater, fake_underwater, training_discriminator)

        # ------ Handle the loss data
        generator_data_handler.append_train_loss(g_loss)
        discriminator_data_handler.append_train_loss(d_loss)

    # ------------------- Validatin the GAN--------------------- #
    # for i, data in enumerate(validation_loader, 0):
    #     with torch.no_grad():
    #         # ------ Evaluation mode
    #         generator.eval()
    #         discriminator.eval()

    #         # ------ Get the data from the data_loader
    #         in_air, underwater = get_data(data, device)

    #         # ------ Evaluate the models
    #         g_loss, fake_underwater = generator.fit(
    #             discriminator, in_air, False)
    #         d_loss = discriminator.fit(underwater, fake_underwater, False)

    #         # ------ Handle the loss data
    #         generator_data_handler.append_valid_loss(g_loss)
    #         discriminator_data_handler.append_valid_loss(d_loss)

    # ---------- Saving images for control
    save_grid(fake_underwater,
              params["output_image"]["saving_path"] + str(epoch), 3)

    # ---------- Handling epoch ending
    g_valid_loss, d_valid_loss = generator_data_handler.custom_multiple_epoch_end(
        epoch, discriminator_data_handler)

    # ---------- Handling model saving
    print("\n------ Saving generator")
    generator_saving_path = params["generator"]["saving_path"].split('.pt')[0]

    torch.save(generator.state_dict(), generator_saving_path +
               "-" + str(epoch) + ".pt")

    # if d_valid_loss == discriminator_data_handler.best_valid_loss:
    #     torch.save(discriminator.state_dict(), params["discriminator"]["saving_path"])
    #     print("\n------ Saving discriminator")

    # ---------- Handling training mode switch
    if (epoch + 1) % params["switch_epochs"] == 0:
        training_generator = not training_generator
        training_discriminator = not training_discriminator

        if training_generator:
            print("\n------ Switching: training generator")
        else:
            print("\n------ Switching: training discriminator")

    generator.print_params()
