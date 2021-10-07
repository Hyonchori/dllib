# DataLoader from LetUIn semiconductor dataset

import torch
import os
import sklearn
import numpy as np
import xmltodict
import albumentations as AT
from PIL import Image


class LetuinDataset(torch.utils.data.Dataset):
    def __init__(self,
                 img_root: str,
                 annot_root: str,
                 img_size: int,
                 stride: int,
                 transform: AT.core.composition.Compose):
        pass


def voc2yolo(annot_root, annot_file, save_root):
    annot_path = os.path.join(annot_root, annot_file)
    with open(annot_path) as f:
        data = f.read()
        data_dict = xmltodict.parse(data)["annotation"]

        img_size = (int(data_dict["size"]["width"]),
                    int(data_dict["size"]["height"]))
        bboxes = data_dict["object"]
        if not isinstance(bboxes, list):
            bboxes = [bboxes]

        txt = ""
        for bbox in bboxes:
            cls = int(bbox["name"].split("_")[-1]) - 1
            bbox = [int(bbox["bndbox"]["xmin"]),
                    int(bbox["bndbox"]["ymin"]),
                    int(bbox["bndbox"]["xmax"]),
                    int(bbox["bndbox"]["ymax"])]
            bbox = xyxy2cpwh(bbox, img_size)
            txt += f"{cls} {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}\n"

        '''save_path = os.path.join(save_root, annot_file.replace("xml", "txt"))
        with open(save_path, "w") as s:
            s.write(txt)'''


def xyxy2cpwh(xyxy, img_size):
    cpwh = [round((xyxy[0] + xyxy[2]) / 2 / img_size[0], 6),
            round((xyxy[1] + xyxy[3]) / 2 / img_size[1], 6),
            round((xyxy[2] - xyxy[0]) / img_size[0], 6),
            round((xyxy[3] - xyxy[1]) / img_size[1], 6)]
    return cpwh



if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from sklearn.model_selection import StratifiedShuffleSplit

    root = "/media/daton/D6A88B27A88B0569/dataset/letuin"
    pattern1_img_root = os.path.join(root, "pattern_1", "pattern_1")
    pattern1_annot_root = os.path.join(root,
                                       "faultdetection_pattern_1_PascalVOC_20210828",
                                       "pattern_type_1")
    pattern2_img_root = os.path.join(root, "pattern_2", "pattern_2")
    pattern2_annot_root = os.path.join(root,
                                       "faultdetection_pattern_2_PascalVOC_20210828",
                                       "pattern_type_2")

    save_root = os.path.join(root, "for_yolo_train", "pattern_1", "labels")
    target_root = pattern2_annot_root
    df_type_dict = {}
    for voc_annot in sorted(os.listdir(target_root)):
        voc2yolo(target_root, voc_annot, save_root)
        name_split = voc_annot.split(".")[0].split("_")[-1].split("-")
        print("\n---")
        print(voc_annot)
        print(name_split)
        df_type = int(name_split[2])
        if df_type not in df_type_dict:
            df_type_dict[df_type] = 1
        else:
            df_type_dict[df_type] += 1
    print(df_type_dict)