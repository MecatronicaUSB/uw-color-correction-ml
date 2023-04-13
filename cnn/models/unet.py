# https://github.com/milesial/Pytorch-UNet/
""" Full assembly of the parts to form the complete network """
import torch.nn as nn
import torch
from .unet_parts import DoubleConv, Down, Up, OutConv


class UNet(nn.Module):
    def __init__(self, params):
        super(UNet, self).__init__()

        n_channels = params["n_channels"]
        bilinear = params["bilinear"]
        learning_rate = params["learning_rate"]
        adam_b1 = params["adam_b1"]
        adam_b2 = params["adam_b2"]

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, 3)
        self.activation_layer = nn.ReLU()

        self.optimizer = torch.optim.Adam(
            self.parameters(), lr=learning_rate, betas=(adam_b1, adam_b2)
        )
        self.loss_function = torch.nn.MSELoss()
        self.lr = learning_rate
        self.saving_path = params["saving_path"] + "unet.pt"

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        out = self.activation_layer(logits)
        return out

    def backpropagate(self, y_pred, y):
        loss, _ = self.calculate_loss(y_pred, y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def calculate_loss(self, y_pred, y):
        loss = self.loss_function(y_pred, y)
        return loss, loss.item()

    def fit(self, image, gt):
        # ------ Reset gradients
        self.optimizer.zero_grad()

        # ------ Generate image
        y_hat = self(image)

        # ------ Backpropagate the UNet
        loss = self.backpropagate(y_hat, gt)

        return y_hat, loss

    def evaluate(self, image, gt):
        # ------ Generate image
        y_hat = self(image)

        # ------ Calculate loss
        _, loss = self.calculate_loss(y_hat, gt)

        return loss

    def save_weights(self):
        torch.save(self.state_dict(), self.saving_path)
