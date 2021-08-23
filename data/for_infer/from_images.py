# Image from image files for inference

from pathlib import Path
import glob
import os
import cv2
import torch
import numpy as np

from dllib.utils.img_utils import letterbox

img_formats = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo']  # acceptable image suffixes


class LoadImages:
    def __init__(self, path, img_size=640, stride=32, auto=True):
        p = str(Path(path).absolute())
        if "*" in p:
            files = sorted(glob.glob(p, recursive=True))
        elif os.path.isdir(p):
            files = sorted(glob.glob(os.path.join(p, "*.*")))
        elif os.path.isfile(p):
            files = [p]
        else:
            raise Exception(f"ERROR: {p} does not exit")

        images = [x for x in files if x.split(".")[-1].lower() in img_formats]
        ni = len(images)

        self.img_size = img_size
        self.stride = stride
        self.auto = auto
        self.files = files
        self.nf = ni
        self.mode = "image"
        assert self.nf > 0, f"No images found in {p}. " \
                            f"Supported formats are:\nimages: {img_formats}"

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count == self.nf:
            raise StopIteration
        path = self.files[self.count]

        self.count += 1
        img0 = cv2.imread(path)
        assert img0 is not None, "Image Not Found " + path
        print(f"image {self.count}/{self.nf} {path}")

        img = letterbox(img0, self.img_size, stride=self.stride, auto=self.auto)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).unsqueeze(0)

        return img0, img, path


if __name__ == "__main__":
    test_path = "/media/daton/D6A88B27A88B0569/dataset/mot/MOT17/train/MOT17-02-DPM/img1"
    dataset = LoadImages(test_path)
    for img0, img, path in dataset:
        print("\n---")
        print(img0.shape)
        print(img.shape)
        print(path)
        cv2.imshow("img0", img0)
        cv2.imshow("img", img[0].numpy().transpose(1, 2, 0))
        cv2.waitKey(1)
