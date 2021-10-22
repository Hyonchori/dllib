
import os
import numpy as np
from PIL import Image

import torch
import cv2
import albumentations as AT

from dllib.utils.img_utils import letterbox


class MilitaryCivilDataset(torch.utils.data.Dataset):
    def __init__(self, dir_root, img_size=96,
                 transform: AT.core.composition.Compose=None):
        self.clss = [x for x in os.listdir(dir_root) if os.path.isdir(os.path.join(dir_root, x))]
        self.nc = len(self.clss)

        self.imgs = []
        for i, cls in enumerate(self.clss):
            imgs = os.listdir(os.path.join(dir_root, cls))
            for img in imgs:
                img_path = os.path.join(dir_root, cls, img)
                self.imgs.append((img_path, i))

        self.img_size = img_size
        self.transform = transform

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        target_img = self.imgs[idx]
        img_path = target_img[0]
        img_label = target_img[1]

        img =  np.array(Image.open(img_path).convert("RGB"))
        label = np.array([0., 0.])
        label[img_label] = 1.

        if self.transform is not None:
            transformed = self.transform(image=img)
            img = transformed["image"]

        img_resize, _, _ = letterbox(img, self.img_size, auto=False)
        img_torch = img_resize[:, :, ::-1].transpose(2, 0, 1)
        img_torch = np.ascontiguousarray(img_torch)
        img_torch = torch.from_numpy(img_torch).unsqueeze(0)

        return img, img_torch, label, img_path


root = "/media/daton/D6A88B27A88B0569/dataset/military-civil"
train_root = os.path.join(root, "train")
valid_root = os.path.join(root, "valid")


def collate_fn(batch):
    img, img_tensor, label, path = zip(*batch)
    return img, torch.cat(img_tensor), np.array(label), path


def get_mc_valid_dataloader(img_size=96, transform=None, batch_size=32):
    dataset = MilitaryCivilDataset(valid_root, img_size=img_size, transform=transform)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )
    return dataloader


def get_mc_train_dataloader(img_size=96, transform=None, batch_size=32):
    dataset = MilitaryCivilDataset(train_root, img_size=img_size, transform=transform)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    return dataloader


if __name__ == "__main__":
    train_dataloader = get_mc_train_dataloader()
    valid_dataloader = get_mc_valid_dataloader()

    for img, img_tensor, label, path in train_dataloader:
        break