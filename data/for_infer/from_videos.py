# Image from video files for inference

import glob
import os
import cv2
import torch
import numpy as np
from pathlib import Path

from dllib.utils.img_utils import letterbox

vid_formats = ['mov', 'avi', 'mp4', 'mpg', 'mpeg', 'm4v', 'wmv', 'mkv']  # acceptable video suffixes


class LoadVideo:
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

        videos = [x for x in files if x.split(".")[-1].lower() in vid_formats]
        nv = len(videos)

        self.img_size = img_size
        self.stride = stride
        self.auto = auto
        self.files = files
        self.nf = nv
        self.mode = "video"
        if any(videos):
            self.new_video(videos[0])
        else:
            self.cap = None
        assert self.nf > 0, f"No images found in {p}. " \
                            f"Supported formats are:\nimages: {vid_formats}"

    def new_video(self, path):
        self.frame = 0
        self.cap = cv2.VideoCapture(path)
        self.nframes = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count == self.nf:
            raise StopIteration
        path = self.files[self.count]

        ret_val, img0 = self.cap.read()
        if not ret_val:
            self.count += 1
            self.cap.release()
            if self.count == self.nf:
                raise StopIteration
            else:
                path = self.files[self.count]
                self.new_video(path)
                ret_val, img0 = self.cap.read()
        self.frame += 1
        print(f"video {self.count + 1}/{self.nf} ({self.frame}/{self.nframes}) {path}")

        img = letterbox(img0, self.img_size, stride=self.stride, auto=self.auto)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).unsqueeze(0)

        return img0, img, path


if __name__ == "__main__":
    test_path = "samples/"
    dataset = LoadVideo(test_path)
    for img0, img, path in dataset:
        print("\n---")
        print(img0.shape)
        print(img.shape)
        print(path)
        cv2.imshow("img0", img0)
        cv2.imshow("img", img[0].numpy().transpose(1, 2, 0))
        cv2.waitKey(1)
