import decorators.validators as validators
from utils.torch_utils import add_channel_first
import torch.nn as nn
import torch
import numpy as np
import sys

sys.path.insert(1, "../")


class Generator(nn.Module):
    # PyTorch: [channels, height, width]
    # @validators.construct_generator
    def __init__(self, params, training=True):
        super(Generator, self).__init__()

        betas_d = params["betas_d"]
        betas_b = params["betas_b"]
        b_c = params["b_c"]
        learning_rate = params["learning_rate"]
        adam_b1 = params["adam_b1"]
        adam_b2 = params["adam_b2"]

        betas_d = torch.tensor([[[[betas_d[0]]], [[betas_d[1]]], [[betas_d[2]]]]])
        betas_b = torch.tensor([[[[betas_b[0]]], [[betas_b[1]]], [[betas_b[2]]]]])
        b_c = torch.tensor([[[[b_c[0]]], [[b_c[1]]], [[b_c[2]]]]])

        self.betas_d = torch.nn.Parameter(betas_d)
        self.betas_b = torch.nn.Parameter(betas_b)
        self.b_c = torch.nn.Parameter(b_c)

        self.lr = learning_rate
        self.optimizer = torch.optim.Adam(
            self.parameters(), lr=learning_rate, betas=(adam_b1, adam_b2)
        )
        self.loss_function = torch.nn.BCELoss()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.training = training
        self.saving_path = params["saving_path"]

    """
      Calculate I_c (this is the distorted image)
          I_c = D_c + B_c
          D_c = J_c * exp(-depth * betas_d_c)
          B_c = b_c * (1 - exp(-depth * betas_b_c))
      
      Where:
          J_c: input in-air image
          depth: distance from object to camera (pixel per pixel)
          b_c, betas_d_c and betas_b_c: NN parameters
    """

    # @validators.generator_forward
    def forward(self, rgb, depth):
        # Calculate exponential values
        D_exp = self.calculate_exp(depth, self.betas_d)
        B_exp = self.calculate_exp(depth, self.betas_b)

        assert rgb.shape == D_exp.shape, "rgb and D_exp must have the same dimensions"
        assert rgb.shape == B_exp.shape, "rgb and B_exp must have the same dimensions"

        # Calculate D_c and B_c
        D_c = rgb * D_exp
        B_c = self.b_c * (1 - B_exp)

        # Calculate I_c
        I_c = D_c + B_c

        return I_c

    @validators.calculate_exp
    def calculate_exp(self, depth, betas):
        return torch.exp(-torch.multiply(depth, betas))

    def backpropagate(self, y_pred, y):
        loss, _ = self.calculate_loss(y_pred, y)

        if self.training:
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

    def fit(self, in_air, fake_prediction):
        # ------ Create valid ground truth
        valid_gt = torch.ones(in_air.shape[0], 1, requires_grad=False).to(self.device)

        # ------ Zero gradients
        self.optimizer.zero_grad()

        # ------ Backpropagate generator
        g_loss = self.backpropagate(fake_prediction, valid_gt)

        return g_loss

    def print_params(self):
        betas_d = self.betas_d.detach().to("cpu").numpy()[0]
        betas_d = np.transpose(betas_d, (1, 2, 0))[0, 0]

        betas_b = self.betas_b.detach().to("cpu").numpy()[0]
        betas_b = np.transpose(betas_b, (1, 2, 0))[0, 0]

        b_c = self.b_c.detach().to("cpu").numpy()[0]
        b_c = np.transpose(b_c, (1, 2, 0))[0, 0]

        print("betas_d: {}".format(betas_d))
        print("betas_b: {}".format(betas_b))
        print("b_c: {}".format(b_c))

    def save_weights(self, epoch):
        saving_path = "{0}generator-{1}.pt".format(self.saving_path, epoch)

        print("Saving generator weights to {0}".format(saving_path))
        torch.save(self.state_dict(), saving_path)
