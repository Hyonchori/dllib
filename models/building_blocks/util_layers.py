
import torch
import torch.nn as nn


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
    def __init__(self, size=None, scale_factor=None, mode="nearest"):
        super().__init__(size, scale_factor, mode)