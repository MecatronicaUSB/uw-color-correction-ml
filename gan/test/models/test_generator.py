import sys
import numpy as np
import torch
import unittest
sys.path.insert(1, '../../')
from models import Generator, Discriminator

class TestGenerator(unittest.TestCase):
    img_size = (1, 4, 480, 640)
    mock_rgbd = torch.tensor(
        [
            [
                [
                    [1, 1],
                    [1, 1],
                ],
                [
                    [1, 1],
                    [1, 1],
                ],
                [
                    [1, 1],
                    [1, 1],
                ],
                [
                    [0, 0],
                    [0, 0],
                ]
            ],
            [
                [
                    [1, 1],
                    [1, 1],
                ],
                [
                    [1, 1],
                    [1, 1],
                ],
                [
                    [1, 1],
                    [1, 1],
                ],
                [
                    [0.5, 0.5],
                    [0.5, 0.5],
                ]
            ],
        ]
    )

    def train_generator_mock(self, generator, discriminator, device='cpu'):
        for i in range(2):
            # ------ Creating in air image
            in_air = np.ones(self.img_size)
            in_air = torch.from_numpy(in_air).float().to(device)

            # ------ Creating ground truth
            valid = torch.tensor(np.ones((in_air.shape[0], 1)), requires_grad=False).float().to(device)

            # ------ Generate fake underwater images
            fake_underwater = generator(in_air)

            # ------ Do a fake prediction
            fake_prediction = discriminator(fake_underwater)

            # ------ Backpropagate the Generator
            g_loss = generator.backpropagate(fake_prediction, valid)
            
            if i == 0:
                initial_loss = g_loss

        return initial_loss, g_loss

    def test_rgbd_split(self):
        generator = Generator(betas=[1., 1., 1.], b_c=1.)
        rgb, depth = generator.split_rgbd(self.mock_rgbd)

        assert rgb.shape == (2, 3, 2, 2), 'RGB must have 2,3,2,2 dimensions'
        assert depth.shape == (2, 1, 2, 2), 'RGB must have 2,1,2,2 dimensions'
        assert torch.all(rgb), 'RGB must only contain ones'
        assert torch.all(depth) == False, 'Depth must only contain zeroes'


    def test_t_calculation_1(self):
        generator = Generator(betas=[1., 1., 1.], b_c=1.)
        _, depth = generator.split_rgbd(self.mock_rgbd)

        expected_t = torch.tensor([[[[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]]],
                                    [[[0.60653065, 0.60653065], [0.60653065, 0.60653065]], 
                                    [[0.60653065, 0.60653065], [0.60653065, 0.60653065]], [[0.60653065, 0.60653065], [0.60653065, 0.60653065]]]])
        result_t = generator.calculate_t(depth, generator.betas)

        assert torch.equal(expected_t, result_t), 'Expected T does not match result T'


    def test_t_calculation_2(self):
        generator = Generator(betas=[0.1, 0.5, 1.], b_c=1.)
        _, depth = generator.split_rgbd(self.mock_rgbd)

        expected_t = torch.tensor([[[[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]]],
                                    [[[0.95122942, 0.95122942], [0.95122942, 0.95122942]], [[0.77880078, 0.77880078], [0.77880078, 0.77880078]], [[0.60653065, 0.60653065], [0.60653065, 0.60653065]]]])
        result_t = generator.calculate_t(depth, generator.betas)

        assert torch.equal(expected_t, result_t), 'Expected T does not match result T'


    def test_t_calculation_3(self):
        generator = Generator(betas=[0.1, 0.5, 0.75], b_c=1.)
        depth = torch.tensor([[[[0.5, 0.4], [0.1, 0.25]]], [[[0.5, 0.4], [0.1, 0.25]]]])

        expected_t = torch.exp(-torch.tensor([[[[0.05, 0.04], [0.01, 0.025]], [[0.25, 0.20], [0.05, 0.125]], [[0.375, 0.3], [0.075, 0.1875]]],
                                            [[[0.05, 0.04], [0.01, 0.025]], [[0.25, 0.20], [0.05, 0.125]], [[0.375, 0.3], [0.075, 0.1875]]]]))
        result_t = generator.calculate_t(depth, generator.betas)

        assert torch.equal(expected_t, result_t), 'Expected T does not match result T'


    def test_d_calculation_1(self):
        generator = Generator(betas=[1., 1., 1.], b_c=1.)
        rgb, _ = generator.split_rgbd(self.mock_rgbd)
        t = torch.tensor([[[[1, 1], [0.5, 0.5]], [[1, 1], [1, 1]], [[1, 1], [0.75, 0.32]]], 
                        [[[1, 1], [0.5, 0.5]], [[1, 1], [1, 1]], [[1, 1], [0.75, 0.32]]]])

        result_d = generator.calculate_d(rgb, t)
        expected_d = torch.tensor([[[[1, 1], [0.5, 0.5]], [[1, 1], [1, 1]], [[1, 1], [0.75, 0.32]]],
                                    [[[1, 1], [0.5, 0.5]], [[1, 1], [1, 1]], [[1, 1], [0.75, 0.32]]]])

        assert torch.equal(expected_d, result_d), 'Expected d does not match result d'

    def test_d_calculation_2(self):
        generator = Generator(betas=[1., 1., 1.], b_c=1.)
        rgb, _ = generator.split_rgbd(self.mock_rgbd)
        t = torch.tensor([[[[0.5, 0.5], [0.5, 0.5]], [[0.75, 0.75], [0.75, 0.75]], [[0.32, 0.32], [0.32, 0.32]]],
                            [[[0.5, 0.5], [0.5, 0.5]], [[0.75, 0.75], [0.75, 0.75]], [[0.32, 0.32], [0.32, 0.32]]]])

        result_d = generator.calculate_d(rgb, t)
        expected_d = torch.tensor([[[[0.5, 0.5], [0.5, 0.5]], [[0.75, 0.75], [0.75, 0.75]], [[0.32, 0.32], [0.32, 0.32]]],
                                    [[[0.5, 0.5], [0.5, 0.5]], [[0.75, 0.75], [0.75, 0.75]], [[0.32, 0.32], [0.32, 0.32]]]])

        assert torch.equal(expected_d, result_d), 'Expected d does not match result d'


    def test_b_calculation_1(self):
        generator = Generator(betas=[1., 1., 1.], b_c=1.)
        t = torch.tensor([[[[1, 1], [0.25, 0.25]], [[1, 1], [1, 1]], [[1, 1], [0, 0]]],
                        [[[1, 1], [0.25, 0.25]], [[1, 1], [1, 1]], [[1, 1], [0, 0]]]])

        result_b = generator.calculate_b(generator.b_c, t)
        expected_b = torch.tensor([[[[0, 0], [0.75, 0.75]], [[0, 0], [0, 0]], [[0, 0], [1, 1]]],
                                    [[[0, 0], [0.75, 0.75]], [[0, 0], [0, 0]], [[0, 0], [1, 1]]]])

        assert torch.equal(expected_b, result_b), 'Expected d does not match result b'


    def test_b_calculation_2(self):
        generator = Generator(betas=[1., 1., 1.], b_c=2.)
        t = torch.tensor([[[[1, 1], [0.25, 0.25]], [[1, 1], [1, 1]], [[1, 1], [0, 0]]],
                            [[[1, 1], [0.25, 0.25]], [[1, 1], [1, 1]], [[1, 1], [0, 0]]]])

        result_b = generator.calculate_b(generator.b_c, t)
        expected_b = torch.tensor([[[[0, 0], [1.5, 1.5]], [[0, 0], [0, 0]], [[0, 0], [2, 2]]],
                                    [[[0, 0], [1.5, 1.5]], [[0, 0], [0, 0]], [[0, 0], [2, 2]]]])

        assert torch.equal(expected_b, result_b), 'Expected d does not match result b'


    def test_decolored_calculation(self):
        generator = Generator(betas=[1., 1., 1.], b_c=1.)

        d = torch.tensor([[[[0, 0], [0.3, 0.2]], [[0, 0], [0, 0]], [[0, 0], [0.5, 0.75]]],
                        [[[0, 0], [0.3, 0.2]], [[0, 0], [0, 0]], [[0, 0], [0.5, 0.75]]]])
        b = torch.tensor([[[[0, 0], [0.75, 0.75]], [[0, 0], [0, 0]], [[0, 0], [1, 1]]],
                            [[[0, 0], [0.75, 0.75]], [[0, 0], [0, 0]], [[0, 0], [1, 1]]]])
        
        expected_decolored = torch.tensor([[[[0, 0], [1.05, 0.95]], [[0, 0], [0, 0]], [[0, 0], [1.5, 1.75]]],
                                            [[[0, 0], [1.05, 0.95]], [[0, 0], [0, 0]], [[0, 0], [1.5, 1.75]]]])
        result_decolored = generator.calculate_decolored(d, b)
        assert torch.equal(expected_decolored, result_decolored), 'Expected decolored does not match result decolored'
        

    def test_forward_1(self):
        generator = Generator(betas=[1., 1., 1.], b_c=1.)

        expected_rgb = torch.tensor([[[[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]]],
                                    [[[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]]]])
        result_rgb = generator(self.mock_rgbd)

        assert torch.equal(expected_rgb, result_rgb), 'Expected RGB does not match result RGB'


    def test_forward_2(self):
        generator = Generator(betas=[0.1, 0.5, 0.75], b_c=0.5)

        mock_rgbd = torch.tensor([[[[1, 1], [1, 1]], [[1, 1], [1, 1]], [[1, 1], [1, 1]], [[0.5, 0.4], [0.1, 0.25]]],
                                [[[1, 1], [1, 1]], [[1, 1], [1, 1]], [[1, 1], [1, 1]], [[0.5, 0.4], [0.1, 0.25]]]])

        expected_rgb = torch.tensor([[[[0.97561467, 0.9803947],
                                    [0.9950249, 0.9876549]],

                                    [[0.88940036, 0.9093654],
                                    [0.97561467, 0.9412484]],

                                    [[0.8436446, 0.87040913],
                                    [0.9638717, 0.91451454]]],
                                    [[[0.97561467, 0.9803947],
                                    [0.9950249, 0.9876549]],

                                    [[0.88940036, 0.9093654],
                                    [0.97561467, 0.9412484]],

                                    [[0.8436446, 0.87040913],
                                    [0.9638717, 0.91451454]]]])
        
        result_rgb = generator(mock_rgbd)
        assert torch.equal(expected_rgb, result_rgb), 'Expected RGB does not match result RGB'


    def test_betas_changing_after_optim(self):
        initial_betas = [1., 1., 1.]
        
        generator = Generator(betas=initial_betas, b_c=1.)
        discriminator = Discriminator()

        initial_loss, g_loss = self.train_generator_mock(generator, discriminator, 'cpu')

        assert initial_loss >= g_loss, 'Loss is not decreasing'
        assert torch.all(torch.tensor(initial_betas) != generator.betas.detach()), 'Betas did not change'

    def test_b_c_changing_after_optim(self):
        initial_b_c = 1.
        
        generator = Generator(betas=[1., 1., 1.], b_c=initial_b_c)
        discriminator = Discriminator()

        initial_loss, g_loss = self.train_generator_mock(generator, discriminator, 'cpu')

        assert initial_loss >= g_loss, 'Loss is not decreasing'
        assert torch.all(torch.tensor(initial_b_c) != generator.b_c.detach()), 'b_c did not change'

    def test_trains_with_cuda(self):
        initial_betas = [1., 1., 1.]
        initial_b_c = 1.

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        generator = Generator(betas=initial_betas, b_c=initial_b_c).to(device)
        discriminator = Discriminator().to(device)

        initial_loss, g_loss = self.train_generator_mock(generator, discriminator, device)

        assert initial_loss >= g_loss, 'Loss is not decreasing'
        assert torch.all(torch.tensor(initial_b_c).to(device) != generator.b_c.detach()), 'b_c did not change'
        assert torch.all(torch.tensor(initial_betas).to(device) != generator.betas.detach()), 'Betas did not change'
        

if __name__ == '__main__':
    unittest.main()