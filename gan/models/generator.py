import torch.nn as nn
import torch
from torch.autograd import Variable
import numpy as np
import sys
sys.path.insert(1, '../')
from utils.torch_utils import add_channel_first
import decorators.validators as validators


class Generator(nn.Module):
    # PyTorch: [channels, height, width]
    @validators.construct_generator
    def __init__(self, betas=[0.3, 0.3, 0.3], b_c=0.5, learning_rate=0.005, adam_b1=0.5, adam_b2=0.999):
        super(Generator, self).__init__()

        betas = torch.tensor([[[[betas[0]]], [[betas[1]]], [[betas[2]]]]])
        b_c = torch.tensor([b_c])

        self.betas = torch.nn.Parameter(betas)
        self.b_c = torch.nn.Parameter(b_c)
        
        self.lr = learning_rate
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate, betas=(adam_b1, adam_b2))
        self.loss_function = torch.nn.BCELoss()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @validators.generator_forward
    def forward(self, rgbd):
        rgb, depth = self.split_rgbd(rgbd)
        
        # t = exp(-depth * beta(lambda))
        t = self.calculate_t(depth, self.betas)

        assert rgb.shape == t.shape, 'rgb and T must have the same dimensions'

        # d = rgb * t
        d = self.calculate_d(rgb, t)

        # b = b_c * (1 - t)
        b = self.calculate_b(self.b_c, t)

        # recolored = b + d
        recolored = self.calculate_decolored(d, b)

        return recolored

    @validators.calculate_t
    def calculate_t(self, depth, betas):
        return torch.exp(-torch.multiply(depth, betas))

    @validators.two_inputs_same_shape
    def calculate_d(self, rgb, t):        
        return rgb * t

    @validators.all_inputs_tensors
    def calculate_b(self, b_c, t):
        return b_c * (1 - t)

    @validators.all_inputs_tensors
    def calculate_decolored(self, d, b):        
        return d + b
    
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
