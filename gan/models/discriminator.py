# GAN Code edited from:
# https://github.com/eriklindernoren/PyTorch-GAN/tree/master/implementations/gan

import torch.nn as nn
import torch
import sys
import numpy as np
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

        self.lr = learning_rate
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate, betas=(adam_b1, adam_b2))
        self.loss_function = torch.nn.BCELoss()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @validators.all_inputs_tensors
    def forward(self, image):
        out = self.model(image)
        out = out.view(out.shape[0], -1)

        return self.adv_layer(out)

    def fit(self, underwater, fake_underwater, training=True):
        # ------ Create valid and fake ground truth
        valid_gt = (
            torch.tensor(np.ones((underwater.shape[0], 1)), requires_grad=False)
            .float()
            .to(self.device)
        )
        fake_gt = (
            torch.tensor(np.zeros((fake_underwater.detach().shape[0], 1)), requires_grad=False)
            .float()
            .to(self.device)
        )

        # ------ Reset gradients
        self.optimizer.zero_grad()

        # ------ Normalize underwater images
        underwater = underwater / 255

        # ------ Calculate real and fake images discriminator loss
        real_loss = self.loss_function(self(underwater), valid_gt)
        fake_loss = self.loss_function(self(fake_underwater.detach()), fake_gt)
        
        # ------ Backpropagate discriminator
        if training:
            real_loss.backward()
            self.optimizer.step()
            fake_loss.backward()
            self.optimizer.step()

        d_loss = (real_loss + fake_loss) / 2

        return d_loss.item()