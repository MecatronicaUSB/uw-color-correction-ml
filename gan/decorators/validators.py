import torch

def construct_generator(func):
    def inner(self, **data):
        for key, value in data.items():
            if key == 'betas':
                assert len(value) == 3, 'length of betas must be 3'
                assert type(value[0]) == float, 'betas[0] must be float'
                assert type(value[1]) == float, 'betas[1] must be float'
                assert type(value[2]) == float, 'betas[2] must be float'
            else:
                assert type(value) == float, '{} must be float'.format(key)

        return func(self, **data)
    return inner


def all_inputs_tensors(func):
    def inner(self, *args):
        for i, arg in enumerate(args):
            assert torch.is_tensor(arg), 'Argument with position {} must be a tensor'.format(i)

        return func(self, *args)
    return inner

@all_inputs_tensors
def calculate_t(func):
    def inner(self, depth, betas):
        assert len(depth.shape) == 4, 'Depth must have batch, 1, h, w dimensions'
        assert depth.shape[1] == 1, 'Depth must have only one channel'
        assert betas.shape == (1, 3, 1, 1), 'Betas must have 1, 3, 1, 1 dimensions'

        return func(self, depth, betas)
    return inner


@all_inputs_tensors
def two_inputs_same_shape(func):
    def inner(self, input1, input2):
        assert input1.shape == input2.shape, 'input1 and input2 must have the same dimensions'

        return func(self, input1, input2)
    return inner


@all_inputs_tensors
def generator_forward(func):
    def inner(self, rgbd):
        assert len(rgbd.shape) == 4, 'rgbd must have 4 dimensions'
        assert rgbd.shape[1] == 4, 'rgbd must have 4 channels'

        return func(self, rgbd)
    return inner