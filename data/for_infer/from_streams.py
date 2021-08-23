# Image from stream sources for inference

import os
import cv2
import time
import torch
import numpy as np
from threading import Thread

from dllib.utils.general_utils import clean_str
from dllib.utils.img_utils import letterbox


class LoadStreams:
    def __init__(self, sources="streams.txt", img_size=640, stride=32, frame_latency=1, auto=True):
        self.mode = "stream"
        self.img_size = img_size
        self.stride = stride
        self.frame_latency = frame_latency
        self.auto = auto

        if os.path.isfile(sources):
            with open(sources, "r") as f:
                sources = [x.strip() for x in f.read().strip().splitlines() if len(x.strip())]
        else:
            sources = [sources]

        n = len(sources)
        self.imgs, self.fps, self.frames, self.threads = [None] * n, [0] * n, [0] * n, [None] * n
        self.sources = [clean_str(x) for x in sources]
        for i, s in enumerate(sources):
            print(f"{i + 1}/{n}: {s}... ", end="")
            s = eval(s) if s.isnumeric() else s
            cap = cv2.VideoCapture(s)
            assert cap.isOpened(), f"Failed to open {s}"
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.fps[i] = max(cap.get(cv2.CAP_PROP_FPS) % 100, 0) or 30.0
            self.frames[i] = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 0) or float("inf")

            _, self.imgs[i] = cap.read()
            self.threads[i] = Thread(target=self.update, args=([i, cap]), daemon=True)
            print(f" success ({w}x{h} at {self.fps[i]:.2f} FPS).")
            self.threads[i].start()
        print("")

        s = np.stack([letterbox(x, self.img_size, stride=self.stride, auto=self.auto)[0].shape for x in self.imgs], 0)
        self.rect = np.unique(s, axis=0).shape[0] == 1
        if not self.rect:
            print("WARNING: Different stream shapes detected. For optimal performance supply similarly-shaped streams.")

    def update(self, i, cap):
        n, f, read = 0, self.frames[i], self.frame_latency
        while cap.isOpened():
            n += 1
            cap.grab()
            if n % read == 0:
                success, im = cap.retrieve()
                self.imgs[i] = im if success else self.imgs[i] * 0
            # time.sleep(1 / self.fps[i])

    def __iter__(self):
        self.count = - 1
        return self

    def __next__(self):
        self.count += 1
        if not all(x.is_alive() for x in self.threads) or cv2.waitKey(1) == ord('q'):
            cv2.destroyAllWindows()
            raise StopIteration

        imgs0 = self.imgs.copy()
        imgs = [letterbox(x, self.img_size, auto=self.auto, stride=self.stride)[0] for x in imgs0]
        imgs = np.stack(imgs, 0)
        imgs = imgs[..., ::-1].transpose((0, 3, 1, 2))
        imgs = np.ascontiguousarray(imgs)
        imgs = torch.from_numpy(imgs)

        return imgs0, imgs, self.sources

    def __len__(self):
        return len(self.sources)


if __name__ == "__main__":
    sources = "0"
    dataset = LoadStreams(sources, frame_latency=1)
    for imgs0, imgs, paths in dataset:
        print("\n---")
        img0 = imgs0[0]
        print(img0.shape)
        print(imgs.shape)
        cv2.imshow("img0", img0)
        cv2.imshow("img", imgs[0].numpy().transpose(1, 2, 0))
        print(paths)