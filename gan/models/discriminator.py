# GAN Code edited from:
# https://github.com/eriklindernoren/PyTorch-GAN/tree/master/implementations/gan

import torch.nn as nn
import torch
import sys
sys.path.insert(1, '../')
import decorators.validators as validators

class Discriminator(nn.Module):
    def __init__(self, in_channels=3, learning_rate=0.0002, adam_b1=0.5, adam_b2=0.999):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, normalization=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels, 64, normalization=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1, bias=False)
        )
        # TODO: remove this hardcoded dimensions: 30, 40
        self.adv_layer = nn.Sequential(nn.Linear(30 * 40, 1), nn.Sigmoid())

        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate, betas=(adam_b1, adam_b2))
        self.loss_function = torch.nn.BCELoss()

    @validators.all_inputs_tensors
    def forward(self, image):
        out = self.model(image)
        out = out.view(out.shape[0], -1)

        return self.adv_layer(out)

    @validators.all_inputs_tensors
    def backpropagate(self, y_pred, y):
        loss, _ = self.calculate_loss(y_pred, y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    @validators.all_inputs_tensors
    def calculate_loss(self, y_pred, y):
        loss = self.loss_function(y_pred, y)
        return loss, loss.item()