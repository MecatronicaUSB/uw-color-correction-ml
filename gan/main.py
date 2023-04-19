import json
import torch
import os

from models import Discriminator, Generator
from datasets import DataLoaderCreator, get_data
from utils import DataHandler, save_demo, create_saving_paths, handle_training_switch
import matplotlib

matplotlib.use("Agg")

DEMO_IMAGES_INDEXES = [0, 1, 2, 3]

# ---------- Opening parameters
with open(os.path.dirname(__file__) + "parameters.json") as path_file:
    params = json.load(path_file)

# ---------- Creating paths for saved images and weights
create_saving_paths(params)

# ---------- Checking for CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- Dataset
data_loader = DataLoaderCreator(params)
training_loader, _ = data_loader.get_loaders()

# ---------- Models. Discriminator starts training first
generator = Generator(params["generator"], training=False).to(device)
discriminator = Discriminator(params["discriminator"], training=True).to(device)

# ---------- Init data handler
gan_handler = DataHandler()

# ---------- Training epochs
for epoch in range(params["epochs"]):
    # ---------- Printing training mode
    print("\n-------------- Epoch: {0} --------------".format(epoch))
    print(
        "\nTraining: {0}".format("Generator" if generator.training else "Discriminator")
    )

    # ------------------- Training the GAN --------------------- #
    for i, data in enumerate(training_loader, 0):
        # ------ Get the data from the data_loader
        in_air, in_air_depth, underwater = get_data(data, device)

        # ------ Generate fake underwater images
        fake_underwater = generator(in_air, in_air_depth)

        # ------ Fit the Generator
        g_loss = generator.fit(in_air, discriminator(fake_underwater))

        # ------ Fit the Discriminator
        d_loss, accuracy_on_real, accuracy_on_fake = discriminator.fit(
            underwater, fake_underwater
        )

        # ------ Handle the loss data
        gan_handler.append_loss(g_loss, "generator")
        gan_handler.append_loss(d_loss, "discriminator")

        # ------ Save accuracy data
        gan_handler.append_accuracy(accuracy_on_real, accuracy_on_fake)

    # ---------- We save demo and weights only if generator is training
    if generator.training:
        # ---------- Saving generator's weights
        generator.save_weights(epoch)

        # ---------- Saving demo images
        generator.eval()
        with torch.no_grad():
            save_demo(
                generator,
                data_loader.dataset,
                DEMO_IMAGES_INDEXES,
                epoch,
                params,
                device,
            )
        generator.train()

    # ---------- Handling epoch ending
    _, _, _, acc_on_fake = gan_handler.epoch_end(epoch)

    # ------------------- Handling training mode switching --------------------- #
    g_training, d_training = handle_training_switch(
        generator, discriminator, acc_on_fake, params
    )

    # ---------- If we need to switch training mode
    generator.train(mode=g_training if g_training is not None else generator.training)
    discriminator.train(
        mode=d_training if d_training is not None else discriminator.training
    )
