import os

import cv2
from tqdm import tqdm

from coco_dataset import COCODataset
from dllib.utils.img_utils import plot_one_box, colors
from dllib.utils.bbox_utils import xywh2xyxy, xywh2cpwh


def extract_person(dataset, target_img_dir, target_label_dir):
    print("")
    for img0, img, bbox0, bbox, img_name in tqdm(dataset):
        xyxy = xywh2xyxy(bbox)
        cpwh = xywh2cpwh(bbox)
        txt = ""
        for b, c in zip(xyxy, cpwh):
            if b[-1] == 1:
                #plot_one_box(b, img0, color=colors(b[-1], True))
                txt += f"{0} {c[0] / img0.shape[1]:.6f} {c[1] / img0.shape[0]:.6f} {c[2] / img0.shape[1]:.6f} {c[3] / img0.shape[0]:.6f}\n"
        img_path = os.path.join(target_img_dir, img_name.split(".")[-2] + ".jpg")
        label_path = os.path.join(target_label_dir, img_name.split(".")[-2] + ".txt")
        cv2.imwrite(img_path, img0[..., ::-1])
        with open(label_path, "w") as f:
            f.write(txt)

        #cv2.imshow("img", img0[..., ::-1])
        #cv2.waitKey(0)


def main():
    data_root = "/media/daton/D6A88B27A88B0569/dataset/coco"
    train_root = os.path.join(data_root, "train2017", "train2017")
    valid_root = os.path.join(data_root, "val2017", "val2017")
    annot_root = os.path.join(data_root, "annotations_trainval2017", "annotations")
    label_file = os.path.join(data_root, "coco_labels91.txt")

    train_dataset = COCODataset(train_root, annot_root, label_file, target_cls="person")
    valid_dataset = COCODataset(valid_root, annot_root, label_file, target_cls="person")

    target_root = os.path.join(data_root, "person_only")

    target_val_img = os.path.join(target_root, "images", "valid")
    target_val_label = os.path.join(target_root, "labels", "valid")
    extract_person(valid_dataset, target_val_img, target_val_label)

    target_train_img = os.path.join(target_root, "images", "train")
    target_train_label = os.path.join(target_root, "labels", "train")
    extract_person(train_dataset, target_train_img, target_train_label)


if __name__ == "__main__":
    main()
