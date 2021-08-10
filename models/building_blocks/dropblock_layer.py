import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
from pathlib import Path
FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[1].as_posix())

from datonlib.models.building_blocks import common

class DropBlock(nn.Module):
    # A regularization method for convolutional neural networks. Generally better than 'dropout'
    # https://arxiv.org/abs/1810.12890
    def __init__(self,
                 drop_prob: float = 0.2,
                 block_size: int = 5):
        super().__init__()
        self.drop_prob = drop_prob
        self.block_size = block_size

    def forward(self, x):
        # x: (batch_size, channels, height, width)
        if not self.training or self.drop_prob == 0.:
            return x
        else:
            gamma = self._compute_gamma(x)
            mask = torch.bernoulli(torch.ones_like(x) * gamma)
            mask = mask.to(x.device)
            block_mask = self._compute_block_mask(mask)
            out = x * block_mask[:, :, :, :]
            out = out * block_mask.numel() / block_mask.sum()
            return out

    def _compute_gamma(self, x):
        gamma = self.drop_prob / (self.block_size ** 2)
        for fs in x.shape[2: ]:
            # fs: feature map size
            gamma *= fs / (fs - self.block_size + 1)
        return gamma

    def _compute_block_mask(self, mask):
        block_mask = F.max_pool2d(input=mask,
                                  kernel_size=(self.block_size, self.block_size),
                                  stride=(1, 1),
                                  padding=self.block_size // 2)
        if self.block_size % 2 == 0:
            block_mask = block_mask[:, :, :-1, :-1]

        block_mask = 1 - block_mask.squeeze(1)
        return block_mask


class DBLinearSheduler(nn.Module):
    def __init__(self,
                 block_size: int,
                 total_steps: int,
                 start_prob: float = 0.,
                 stop_prob: float = 0.2,
                 ):
        super().__init__()
        self.dropblock = DropBlock(start_prob, block_size)
        self.tmp_step = 0
        self.drop_values = np.linspace(start=start_prob, stop=stop_prob, num=total_steps)

    def forward(self, x, epoch=None):
        if epoch is not None:
            idx = min(epoch, len(self.drop_values))
            self.dropblock.drop_prob = self.drop_values[idx]
        return self.dropblock(x)

    def step(self):
        if self.tmp_step < len(self.drop_values):
            self.dropblock.drop_prob = self.drop_values[self.tmp_step]

        self.tmp_step += 1

import time
import matplotlib.pyplot as plt

if __name__ == "__main__":
    sample = torch.randn(2, 3, 256, 256)

    dropblock = DropBlock(0.5, 5)

    t1 = time.time()
    dropped_sample = dropblock(sample)
    print(time.time() - t1)

    print(dropped_sample.size())

    dbs = DBLinearSheduler(5, 100)
    print(dbs.drop_values)

    for i in range(100):
        dropped_sample = dbs(sample, epoch=i)
        #dbs.step()
        print(dbs.dropblock.drop_prob)