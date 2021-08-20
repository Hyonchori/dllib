# DataLoader from fire dataset by AIhub

import torch
import os
import numpy as np
from PIL import Image
import albumentations as AT

from dllib.utils.img_utils import colors, letterbox, plot_one_box
from dllib.utils.bbox_utils import xywh2xyxy, letterboxed_xywh


class FireDataset(torch.utils.Dataset):
    def __init__(self):
        pass


if __name__ == "__main__":
    root =