
import torch
import torch.nn as nn


class ConcatLayer(nn.Module):
    # Implementation of torch.cat() in layer
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension

    def forward(self, xs):
        # xs = iterable of torch.tensor: (tensor, tensor, ...) or [tensor, tensor, ...]
        return torch.cat(xs, self.d)