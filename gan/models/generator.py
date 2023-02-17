import decorators.validators as validators
from utils.torch_utils import add_channel_first
import torch.nn as nn
import torch
from torch.autograd import Variable
import numpy as np
import sys
sys.path.insert(1, '../')


class Generator(nn.Module):
    # PyTorch: [channels, height, width]
    @validators.construct_generator
    def __init__(self, params):
        super(Generator, self).__init__()

        betas_d = params["betas_d"]
        betas_b = params["betas_b"]
        b_c = params["b_c"]
        learning_rate = params["learning_rate"]
        adam_b1 = params["adam_b1"]
        adam_b2 = params["adam_b2"]

        betas_d = torch.tensor(
            [[[[betas_d[0]]], [[betas_d[1]]], [[betas_d[2]]]]])
        betas_b = torch.tensor(
            [[[[betas_b[0]]], [[betas_b[1]]], [[betas_b[2]]]]])
        b_c = torch.tensor([b_c])

        self.betas_d = torch.nn.Parameter(betas_d)
        self.betas_b = torch.nn.Parameter(betas_b)
        self.b_c = torch.nn.Parameter(b_c)

        self.lr = learning_rate
        self.optimizer = torch.optim.Adam(
            self.parameters(), lr=learning_rate, betas=(adam_b1, adam_b2))
        self.loss_function = torch.nn.BCELoss()
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

    '''
      Calculate I_c (this is the distorted image)
          I_c = D_c + B_c
          D_c = J_c * exp(-depth * betas_d_c)
          B_c = b_c * (1 - exp(-depth * betas_b_c))
      
      Where:
          J_c: input in-air image
          depth: distance from object to camera (pixel per pixel)
          b_c, betas_d_c and betas_b_c: NN parameters
    '''
    @validators.generator_forward
    def forward(self, rgbd):
        rgb, depth = self.split_rgbd(rgbd)

        # Normalize rgb and depth input
        rgb = rgb / 255
        depth = depth / 10

        # Calculate exponential values
        D_exp = self.calculate_exp(depth, self.betas_d_c)
        B_exp = self.calculate_exp(depth, self.betas_d_c)

        assert rgb.shape == D_exp.shape, 'rgb and D_exp must have the same dimensions'
        assert rgb.shape == B_exp.shape, 'rgb and B_exp must have the same dimensions'

        # Calculate D_c and B_c
        D_c = rgb * D_exp
        B_c = self.b_c * (1 - B_exp)

        # Calculate I_c
        I_c = D_c + B_c

        # Clamp the value from 0 to 1
        return torch.clamp(I_c, min=0, max=1)

    @validators.calculate_exp
    def calculate_exp(self, depth, betas):
        return torch.exp(-torch.multiply(depth, betas))

    def backpropagate(self, y_pred, y, backpropagate=True):
        loss, _ = self.calculate_loss(y_pred, y)

        if backpropagate:
            loss.backward()
            self.optimizer.step()

        return loss.item()

    @validators.all_inputs_tensors
    def calculate_loss(self, y_pred, y):
        loss = self.loss_function(y_pred, y)
        return loss, loss.item()

    @validators.generator_forward
    def split_rgbd(self, rgbd):
        rgb = rgbd[:, :3, :, :]
        depth = add_channel_first(rgbd[:, 3, :, :])

        return rgb, depth

    def fit(self, discriminator, in_air, training=True):
        # ------ Create valid ground truth
        valid_gt = (
            torch.tensor(np.ones((in_air.shape[0], 1)), requires_grad=False)
            .float()
            .to(self.device)
        )

        # ------ Reset gradients
        self.optimizer.zero_grad()

        # ------ Generate fake underwater images
        fake_underwater = self(in_air)

        # ------ Do a fake prediction
        fake_prediction = discriminator(fake_underwater)

        # ------ Backpropagate generator
        g_loss = self.backpropagate(fake_prediction, valid_gt, training)

        return g_loss, fake_underwater
