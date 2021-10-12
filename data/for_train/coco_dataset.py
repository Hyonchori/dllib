# DataLoader from COCO dataset

import torch
import os
import numpy as np
from PIL import Image
import albumentations as AT

from dllib.utils.img_utils import colors, letterbox, plot_one_box, plot_one_keypoint, crop_bbox
from dllib.utils.bbox_utils import xywh2xyxy, letterboxed_xywh, letterboxed_keypoint, \
                                normalize_xyxy, normalize_keypoint, xywh2cpwh, unnormalize_keypoint

max_keypoint_det = 112
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

        if mode not in ["detection", "semantice_seg", "instance_seq", "keypoint", "only keypoint"]:
            raise Exception(f"'mode' should be selected between \
            ['detection', 'semantic_seg', 'instance_seg', 'keypoint', 'only keypoint']")
        self.mode = mode

        target_annot = None
        if target_trainval is not None:
            if mode == "detection" or "seg" in mode:
                target_annot = os.path.join(annot_root, "instances_{}{}.json".format(target_trainval, year))
            elif "keypoint" in mode:
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
        if self.mode == "only keypoint":
            self.crop_img_size = img_size
        else:
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
                            "keypoint": self.keypoint_collate_fn,
                            "only keypoint": self.only_keypoint_collate_fn}

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        if self.mode == "detection":
            return self.get_detection_item(idx)
        elif self.mode == "semantic_seg":
            return self.get_semantic_seg_item(idx)
        elif self.mode == "instance_seg":
            return self.get_instance_seg_item(idx)
        elif "keypoint" in self.mode:
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

        if len(bboxes0) == 0:
            bboxes0 = [bboxes0]
            bboxes = [bboxes]

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

        if len(bboxes0) == 0:
            bboxes0 = [bboxes0]
            bboxes = [bboxes]

        if len(keypoints0) == 0:
            keypoints0 = [keypoints0]
            keypoints = [keypoints]

        bboxes0 = np.stack(bboxes0)
        bboxes = torch.from_numpy(np.stack(bboxes))

        keypoints0 = np.stack(keypoints0)
        keypoints = torch.from_numpy(np.stack(keypoints))

        return img0, img, bboxes0, bboxes, keypoints0, keypoints, img_name

    def detection_collate_fn(self, batch):
        img0, img, bboxes0, bboxes, img_name = zip(*batch)
        img_b, bbox_b = [], []
        for im, bbox, name in zip(img, bboxes, img_name):
            if len(bbox[0]) == 0:
                continue
            im, ratio, (dw, dh) = letterbox(im, self.img_size, auto=False, stride=self.stride)
            im = im[:, :, ::-1].transpose(2, 0, 1)
            im = np.ascontiguousarray(im)
            im = torch.from_numpy(im).unsqueeze(0)
            print(name, bbox)
            bbox = letterboxed_xywh(bbox, ratio, dw, dh)
            img_b.append(im)
            bbox_b.append(bbox)
        return img0, torch.cat(img_b), bboxes0, bbox_b, img_name

    def keypoint_collate_fn(self, batch):
        img0, img, bboxes0, bboxes, keypoints0, keypoints, img_name = zip(*batch)
        img_b, bbox_b, keypoint_b = [], [], []
        for im, bbox, keypoint in zip(img, bboxes, keypoints):
            if sum(keypoint[0]) == 0:
                continue
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

    def only_keypoint_collate_fn(self, batch):
        img0, img, bboxes0, bboxes, keypoints0, keypoints, img_name = zip(*batch)
        img_b, keypoint_b, img_name_b = [], [], []
        for i, (im, bbox, keypoint) in enumerate(zip(img, bboxes, keypoints)):
            if sum(keypoint[0]) == 0:
                continue
            xyxy = self.xywh2xyxy(bbox)
            cropped_img, adj_keypoints = crop_bbox(im, xyxy, keypoint, self.crop_img_size, normalize=False)
            img_b.append(torch.from_numpy(np.ascontiguousarray(cropped_img[:, :, :, ::-1].transpose(0, 3, 1, 2))))
            keypoint_b.append(torch.from_numpy(adj_keypoints))
            img_name_b.append(img_name[i])

        img_b = torch.cat(img_b)
        img_b = img_b[: max_keypoint_det]

        keypoint_b = torch.cat(keypoint_b)
        keypoint_b = keypoint_b[: max_keypoint_det]
        return img_b, keypoint_b, img_name_b

    @staticmethod
    def xywh2xyxy(xywhs):
        xyxy = xywhs.copy()
        xyxy[:, 2] = xywhs[:, 0] + xywhs[:, 2]
        xyxy[:, 3] = xywhs[:, 1] + xywhs[:, 3]
        return xyxy


data_root = "/media/daton/D6A88B27A88B0569/dataset/coco"
train_root = os.path.join(data_root, "train2017", "train2017")
valid_root = os.path.join(data_root, "val2017", "val2017")
annot_root = os.path.join(data_root, "annotations_trainval2017", "annotations")
label_file = os.path.join(data_root, "coco_labels91.txt")
keypoint_label_file = os.path.join(data_root, "coco_keypoint_labels.txt")
#keypoint_label_file = None
train_transform = AT.Compose([
    AT.ColorJitter(),
    AT.HueSaturationValue(),
    AT.RandomBrightnessContrast(),
    #AT.Normalize(),
    AT.Cutout(max_h_size=16, max_w_size=16)
])


def get_coco2017_valid_dataloader(img_size: (int, int),
                                  mode: str="detection",
                                  target_cls: str=None,
                                  batch_size: int=16,
                                  transform : AT.core.composition.Compose=None):
    dataset = COCODataset(valid_root, annot_root, target_cls=target_cls, mode=mode, img_size=img_size,
                          labels=label_file, keypoint_labels=keypoint_label_file,
                          transform=transform)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=dataset.collate_fns[mode]
    )
    return dataloader


def get_coco2017_train_dataloader(img_size: (int, int),
                                  mode: str="detection",
                                  target_cls: str=None,
                                  batch_size: int=16,
                                  transform : AT.core.composition.Compose=None):
    dataset = COCODataset(train_root, annot_root, target_cls=target_cls, mode=mode, img_size=img_size,
                          labels=label_file, keypoint_labels=keypoint_label_file,
                          transform=transform)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=dataset.collate_fns[mode]
    )
    return dataloader


def get_coco2017dataloader(img_size: (int, int),
                           mode: str="detection",
                           target_cls=None,
                           train_batch: int=32,
                           valid_batch: int=16,
                           train_transform: AT.core.composition.Compose=None,
                           valid_transform: AT.core.composition.Compose=None):
    train_dataloader = get_coco2017_train_dataloader(img_size, mode, target_cls, train_batch, train_transform)
    valid_dataloader = get_coco2017_valid_dataloader(img_size, mode, target_cls, valid_batch, valid_transform)
    return train_dataloader, valid_dataloader


if __name__ == "__main__":
    import cv2

    '''# only keypoint dataloader
    valid_dataloader = get_coco2017_valid_dataloader(img_size=128,
                                                     mode="only keypoint",
                                                     batch_size=16,
                                                     transform=train_transform)
    for img, keypoint_b, img_name in valid_dataloader:
        print("\n#####################")
        print(img.shape, keypoint_b.shape)

        for i, (im, keypoint) in enumerate(zip(img, keypoint_b)):
            print("\n---")
            print(im.shape)
            print(keypoint.shape)
            print(keypoint)

            imm = im.numpy().transpose(1, 2, 0).copy()
            plot_one_keypoint(keypoint, imm)
            cv2.imshow("imm", imm)
            cv2.waitKey(0)
        break'''

    val_dataloader = get_coco2017_valid_dataloader(img_size=640,
                                                   mode="detection",
                                                   target_cls="person",
                                                   batch_size=16)
    for img0, img, bbox0, bbox, name in val_dataloader:
        print("\n####################")
        print(img.shape, len(bbox))

        for i in range(img.shape[0]):
            print("\n---")
            print(name[i])
            print(bbox[i])
            xyxy = xywh2xyxy(bbox[i])
            imm = img[i].numpy().transpose(1, 2, 0).copy()
            #imm = img0[i]

            for b in xyxy:
                if b[-1] == 1:
                    plot_one_box(b, imm, color=colors(b[-1], True))
            cv2.imshow("imm", imm)
            cv2.waitKey(0)
