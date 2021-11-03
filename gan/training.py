import torch
import numpy as np

def train(generator, discriminator, in_air, underwater, loss_function, device):
    # ------- Training mode
    generator.train()
    discriminator.train()

    # ------ Create valid and fake ground truth
    valid = torch.tensor(np.ones((in_air.shape[0], 1)), requires_grad=False).float().to(device)
    fake = torch.tensor(np.zeros((in_air.shape[0], 1)), requires_grad=False).float().to(device)

    # ---------------- Train generator ---------------- #
    generator.optimizer.zero_grad()

    # ------ Generate fake underwater images
    fake_underwater = generator(in_air)

    # ------ Do a fake prediction
    fake_prediction = discriminator(fake_underwater)

    # ------ Backpropagate the Generator
    g_loss = generator.backpropagate(fake_prediction, valid)


    # -------------- Train discriminator -------------- #
    discriminator.optimizer.zero_grad()

    # ------ Calculate real and fake images discriminator loss
    real_loss = loss_function(discriminator(underwater), valid)
    fake_loss = loss_function(discriminator(fake_underwater.detach()), fake)
    d_loss = (real_loss + fake_loss) / 2

    # ------ Backpropagate the Discriminator
    d_loss.backward()
    discriminator.optimizer.step()
    

    return fake_underwater, g_loss, d_loss


def validate_d(generator, discriminator, in_air, underwater, loss_function, device):
    with torch.no_grad():
        generator.eval()
        discriminator.eval()
    
        # ------ Create valid and fake ground truth
        valid = torch.tensor(np.ones((in_air.shape[0], 1)), requires_grad=False).float().to(device)
        fake = torch.tensor(np.zeros((in_air.shape[0], 1)), requires_grad=False).float().to(device)

        # ------ Generate fake underwater images
        fake_underwater = generator(in_air)

        # ------ Calculate real and fake images discriminator loss
        real_loss = loss_function(discriminator(underwater), valid)
        fake_loss = loss_function(discriminator(fake_underwater.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2
    
        return d_loss


def get_data(data, device):
    in_air = data['in_air'].to(device)
    underwater = data['underwater'].to(device)

    return in_air, underwater