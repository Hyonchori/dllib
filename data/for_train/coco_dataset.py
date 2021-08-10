# DataLoader from COCO dataset

import torch
import os
import numpy as np
from PIL import Image
import albumentations as AT

from dllib.utils.img_utils import colors, letterbox, plot_one_box, plot_one_keypoint
from dllib.utils.bbox_utils import xywh2xyxy, letterboxed_xywh, letterboxed_keypoint, \
                                normalize_xyxy, normalize_keypoint


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

        self.collate_fns = {"detection": self.detection_collate_fn,
                            "keypoint": self.keypoint_collate_fn}

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
        img = img0.copy()

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

        bboxes0 = np.stack(bboxes0)
        bboxes = torch.from_numpy(np.stack(bboxes))

        return img0, img, bboxes0, bboxes, img_name

    def get_keypoint_item(self, idx):
        idx = self.img_ids[idx]
        img_name = self.coco.loadImgs(idx)[0]["file_name"]
        img_path = os.path.join(self.img_root, img_name)
        img0 = np.array(Image.open(img_path).convert("RGB"))
        img = img0.copy()

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

        bboxes0 = np.stack(bboxes0)
        bboxes = torch.from_numpy(np.stack(bboxes))

        keypoints0 = np.stack(keypoints0)
        keypoints = torch.from_numpy(np.stack(keypoints))

        return img0, img, bboxes0, bboxes, keypoints0, keypoints, img_name

    def detection_collate_fn(self, batch):
        img0, img, bboxes0, bboxes, img_name = zip(*batch)
        img_b, bbox_b = [], []
        for im, bbox in zip(img, bboxes):
            im, ratio, (dw, dh) = letterbox(im, self.img_size, auto=False, stride=self.stride)
            im = im[:, :, ::-1].transpose(2, 0, 1)
            im = np.ascontiguousarray(im)
            im = torch.from_numpy(im).unsqueeze(0)

            bbox = letterboxed_xywh(bbox, ratio, dw, dh)
            img_b.append(im)
            bbox_b.append(bbox)
        return img0, torch.cat(img_b), bboxes0, bbox_b, img_name

    def keypoint_collate_fn(self, batch):
        img0, img, bboxes0, bboxes, keypoints0, keypoints, img_name = zip(*batch)
        img_b, bbox_b, keypoint_b = [], [], []
        for im, bbox, keypoint in zip(img, bboxes, keypoints):
            im, ratio, (dw, dh) = letterbox(im, self.img_size, auto=False, stride=self.stride)
            im = im[:, :, ::-1].transpose(2, 0, 1)
            im = np.ascontiguousarray(im)
            im = torch.from_numpy(im).unsqueeze(0)

            bbox = letterboxed_xywh(bbox, ratio, dw, dh)
            keypoint = letterboxed_keypoint(keypoint, ratio, dw, dh)
            img_b.append(im)
            bbox_b.append(bbox)
            keypoint_b.append(keypoint)
        return img0, torch.cat(img_b), bboxes0, bbox_b, keypoints0, keypoint_b, img_name


data_root = "/media/daton/D6A88B27A88B0569/dataset/coco"
train_root = os.path.join(data_root, "train2017", "train2017")
valid_root = os.path.join(data_root, "val2017", "val2017")
annot_root = os.path.join(data_root, "annotations_trainval2017", "annotations")
label_file = "coco_labels.txt"
keypoint_label_file = "coco_keypoint_labels.txt"


def get_coco2017dataloader(img_size: (int, int),
                           mode: str="detection",
                           target_cls=None,
                           train_batch: int=32,
                           valid_batch: int=16,
                           train_transform: AT.core.composition.Compose=None,
                           valid_transform: AT.core.composition.Compose=None):
    train_dataset = COCODataset(train_root, annot_root, target_cls=target_cls, mode=mode, img_size=img_size,
                                labels=label_file, keypoint_labels=keypoint_label_file,
                                transform=train_transform)
    valid_dataset = COCODataset(valid_root, annot_root, target_cls=target_cls, mode=mode, img_size=img_size,
                                labels=label_file, keypoint_labels=keypoint_label_file,
                                transform=valid_transform)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_batch,
        shuffle=True,
        collate_fn=train_dataset.collate_fns[mode]
    )
    valid_dataloader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=valid_batch,
        shuffle=False,
        collate_fn=valid_dataset.collate_fns[mode]
    )
    return train_dataloader, valid_dataloader


if __name__ == "__main__":
    import cv2

    train_dataloader, valid_dataloader = get_coco2017dataloader(
        img_size=412, mode="detection", target_cls="person"
    )

    for img0, img_b, bboxes0, bbox_b, img_name in train_dataloader:
        print("\n================")
        print(img_b.shape)
        for img, bbox in zip(img_b, bbox_b):
            _, w, h = img.size()
            xyxy = xywh2xyxy(bbox)
            xyxy_n = normalize_xyxy(xyxy, w, h)
        break


    '''dataset = COCODataset(img_root, annot_root,
                          labels=label_file, keypoint_labels=keypoint_label_file,
                          target_cls="person", mode="keypoint")
    labels = dataset.labels

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        collate_fn=dataset.keypoint_collate_fn
    )

    for img0, img_b, bboxes0, bbox_b, keypoints0, keypoint_b, img_name in dataloader:
        print("\n================")
        print(img_b.shape)
        for img, bbox, keypoint in zip(img_b, bbox_b, keypoint_b):
            print("")
            _, w, h = img.size()
            xyxy = xywh2xyxy(bbox)
            xyxy_n = normalize_xyxy(xyxy, w, h)
            keypoint_n = normalize_keypoint(keypoint.float(), w, h)
            print(keypoint)
            print(keypoint_n)
        break'''


    '''for img0, img, bboxes0, bboxes, keypoints0, keypoints, img_name in dataset:
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
'''