
import torch
import torch.nn as nn


class SEModule(nn.Module):
    # Squeeze and excitation module. arXiv:1709.01507
    def __init__(self, c_in, reduction=16):
        super(SEModule, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(c_in, c_in//reduction, bias=False),
            nn.Mish(),
            nn.Linear(c_in//reduction, c_in, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch_size, c, _, _ = x.size()
        y = self.squeeze(x).view(batch_size, c)
        y = self.excitation(y).view(batch_size, c, 1, 1)
        return x * y.expand_as(x)