
import torch
import torch.nn as nn

from dllib.models.building_blocks.conv_layers import autopad


class ConcatLayer(nn.Module):
    # Implementation of torch.cat() in layer
    def __init__(self, dimension=1, target_layers=[-1]):
        super().__init__()
        self.d = dimension
        self.target_layers = target_layers

    def forward(self, xs):
        # xs = iterable of torch.tensor: (tensor, tensor, ...) or [tensor, tensor, ...]
        return torch.cat(xs, self.d)


class Upsample(nn.Upsample):
    def __init__(self, size=None, scale_factor=None, mode="nearest", target_layer=-1):
        super().__init__(size, scale_factor, mode)
        self.target_layer = target_layer


class Downsample(nn.MaxPool2d):
    def __init__(self, kernel_size, target_layer=-1, stride=None,
                 padding=0, dilation=1, return_indices=False, ceil_mode=False):
        super().__init__(kernel_size, stride, autopad(kernel_size), dilation, return_indices, ceil_mode)
        self.target_layer = target_layer


class GetLayer(nn.Module):
    def __init__(self, target_layer=-1):
        super().__init__()
        self.target_layer = target_layer

