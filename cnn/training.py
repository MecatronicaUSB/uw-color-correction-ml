import torch
import numpy as np

def train(model, image, gt, loss_function, device):
    # ------- Training mode
    model.train()

    # ---------------- Train UNet ---------------- #
    model.optimizer.zero_grad()

    # ------ Generate fake underwater images
    y_hat = model(image)

    # ------ Backpropagate the UNet
    loss = model.backpropagate(y_hat, gt)
    
    return y_hat, loss

def get_data(data, device):
    image, gt = data

    image = image.to(device) / 255
    gt = gt.to(device) / 255

    return image, gt