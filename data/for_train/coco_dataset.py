# DataLoader from COCO dataset

import torch
import os
import numpy as np
from PIL import Image
import albumentations as AT

from dllib.utils.img_utils import colors, letterbox, plot_one_box, plot_one_keypoint
from dllib.utils.bbox_utils import xywh2xyxy, letterboxed_xywh, letterboxed_keypoint


class COCODataset(torch.utils.data.Dataset):
    def __init__(self,
                 img_root: str,
                 annot_root: str=None,
                 labels: str=None,
                 keypoint_labels: str=None,
                 target_cls: str=None,
                 mode: str="detection",
                 img_size: int=640,
                 stride: int=32,
                 transform: AT.core.composition.Compose=None):
        from pycocotools.coco import COCO

        img_dir = img_root.split("/")[-1]
        year = img_dir[-4: ]
        if year not in annot_root:
            raise Exception(f"img and annotation year are not matching!")

        target_trainval = None
        if "train" in img_dir:
            target_trainval = "train"
        elif "val" in img_dir:
            target_trainval = "val"

        if mode not in ["detection", "semantice_seg", "instance_seq", "keypoint"]:
            raise Exception(f"'mode' should be selected between ['detection', 'semantic_seg', 'instance_seg', 'keypoint']")
        self.mode = mode

        target_annot = None
        if target_trainval is not None:
            if mode == "detection" or "seg" in mode:
                target_annot = os.path.join(annot_root, "instances_{}{}.json".format(target_trainval, year))
            elif mode == "keypoint":
                target_annot = os.path.join(annot_root, "person_keypoints_{}{}.json".format(target_trainval, year))
        print(target_annot)

        if os.path.isfile(target_annot):
            self.coco = COCO(target_annot)
        else:
            raise Exception(f"annotations file not found")

        self.img_ids = self.coco.getImgIds() if target_cls is None \
            else self.coco.getImgIds(catIds=self.coco.getCatIds(catNms=target_cls))
        self.transform = transform
        self.img_root = img_root
        self.annot_root = annot_root
        self.img_size = img_size
        self.stride = stride

        self.labels = None
        if labels is not None:
            with open(labels) as f:
                data = f.readlines()
                self.labels = {i: data[i-1][:-1] for i in range(1, len(data)+1)}
        self.keypoint_labels = None
        if keypoint_labels is not None:
            with open(keypoint_labels) as f:
                data = f.readlines()
                self.keypoint_labels = {i: data[i - 1][:-1] for i in range(1, len(data)+1)}

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        if self.mode == "detection":
            return self.get_detection_item(idx)
        elif self.mode == "semantic_seg":
            return self.get_semantic_seg_item(idx)
        elif self.mode == "instance_seg":
            return self.get_instance_seg_item(idx)
        elif self.mode == "keypoint":
            return self.get_keypoint_item(idx)
        else:
            raise Exception(f"'{self.mode}' is invalid mode!")

    def get_detection_item(self, idx):
        idx = self.img_ids[idx]
        img_name = self.coco.loadImgs(idx)[0]["file_name"]
        img_path = os.path.join(self.img_root, img_name)
        img0 = np.array(Image.open(img_path).convert("RGB"))
        img, ratio, (dw, dh) = letterbox(img0, self.img_size, stride=self.stride)

        ann_ids = self.coco.getAnnIds(imgIds=idx)
        anns = self.coco.loadAnns(ann_ids)
        bboxes0, bboxes = [], []
        for ann in anns:
            cls = ann["category_id"]
            x_min, y_min, width, height = ann["bbox"]
            bboxes0.append([x_min, y_min, width, height, 1, cls])  # [x_min, y_min, width, height, conf, cls]
            bboxes.append([x_min, y_min, width, height, 1, cls])

        if self.transform is not None:
            transformed = self.transform(image=img, bboxes=bboxes)
            img = transformed["image"]
            bboxes = transformed["bboxes"]

        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).unsqueeze(0)

        bboxes0 = np.stack(bboxes0)
        bboxes = torch.from_numpy(np.stack(bboxes))
        bboxes = letterboxed_xywh(bboxes, ratio, dw, dh)

        return img0, img, bboxes0, bboxes, img_name


    def get_keypoint_item(self, idx):
        idx = self.img_ids[idx]
        img_name = self.coco.loadImgs(idx)[0]["file_name"]
        img_path = os.path.join(self.img_root, img_name)
        img0 = np.array(Image.open(img_path).convert("RGB"))
        img, ratio, (dw, dh) = letterbox(img0, self.img_size, stride=self.stride)

        ann_ids = self.coco.getAnnIds(imgIds=idx)
        anns = self.coco.loadAnns(ann_ids)
        bboxes0, bboxes = [], []
        keypoints0, keypoints = [], []
        for ann in anns:
            cls = ann["category_id"]
            x_min, y_min, width, height = ann["bbox"]
            bboxes0.append([x_min, y_min, width, height, 1, cls])
            bboxes.append([x_min, y_min, width, height, 1, cls])
            keypoint = ann["keypoints"]
            keypoints0.append(keypoint)
            keypoints.append(keypoint)

        if self.transform is not None:
            transformed = self.transform(image=img, bboxes=bboxes, keypoints=keypoints)
            img = transformed["image"]
            bboxes = transformed["bboxes"]
            keypoints = transformed["keypoints"]

        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).unsqueeze(0)

        bboxes0 = np.stack(bboxes0)
        bboxes = torch.from_numpy(np.stack(bboxes))
        bboxes = letterboxed_xywh(bboxes, ratio, dw, dh)

        keypoints0 = np.stack(keypoints0)
        keypoints = torch.from_numpy(np.stack(keypoints))
        keypoints = letterboxed_keypoint(keypoints, ratio, dw, dh)

        return img0, img, bboxes0, bboxes, keypoints0, keypoints, img_name


if __name__ == "__main__":
    import cv2

    data_root = "/media/daton/D6A88B27A88B0569/dataset/coco"
    img_root = os.path.join(data_root, "val2017", "val2017")
    annot_root = os.path.join(data_root, "annotations_trainval2017", "annotations")
    label_file = "coco_labels.txt"
    keypoint_label_file = "coco_keypoint_labels.txt"
    dataset = COCODataset(img_root, annot_root,
                          labels=label_file, keypoint_labels=keypoint_label_file,
                          target_cls="person", mode="keypoint")
    labels = dataset.labels

    for img0, img, bboxes0, bboxes, keypoints0, keypoints, img_name in dataset:
        print("\n---")
        print(img0.shape)
        print(img.shape)
        print(bboxes0.shape, bboxes.shape)
        print(keypoints0.shape, keypoints.shape)
        print(img_name)

        for *xywh, conf, cls in bboxes0:
            xyxy = xywh2xyxy(torch.tensor(xywh).view(1, 4)).view(-1)
            c = int(cls)
            label = labels[c]
            plot_one_box(xyxy, img0, label=label, color=colors(c, True))

        for kp in keypoints0:
            plot_one_keypoint(kp, img0)



        img_lt = np.ascontiguousarray(img[0].numpy().transpose(1, 2, 0))
        for *xywh, conf, cls in bboxes:
            xyxy = xywh2xyxy(torch.tensor(xywh).view(1, 4)).view(-1)
            c = int(cls)
            label = labels[c]
            plot_one_box(xyxy, img_lt, label=label, color=colors(c, True))

        for kp in keypoints:
            plot_one_keypoint(kp, img_lt)

        cv2.imshow("img0", img0)
        cv2.imshow("img", img_lt)
        cv2.waitKey(0)
