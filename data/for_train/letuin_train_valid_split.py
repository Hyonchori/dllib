# DataLoader from LetUIn semiconductor dataset

import os
import shutil
import xmltodict


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


def get_num_per_type(target_root):
    num_dict = {}
    for annot in sorted(os.listdir(target_root)):
        name_split = annot.split(".")[0].split("_")[-1].split("-")
        df_type = int(name_split[2])
        if df_type not in num_dict:
            num_dict[df_type] = 1
        else:
            num_dict[df_type] += 1
    return num_dict


def split_valid(target_root, num_dict, valid_prob=0.2):
    valid_annot = []
    train_annot = []
    for annot in sorted(os.listdir(target_root)):
        name_split = annot.split(".")[0].split("_")[-1].split("-")
        df_type = int(name_split[2])
        df_num = int(name_split[3])
        if df_num >= int(num_dict[df_type] * (1 - valid_prob)):
            valid_annot.append(annot.replace(".xml", ""))
        else:
            train_annot.append(annot.replace(".xml", ""))
    return train_annot, valid_annot


def mv_train_valid(target_root, train, valid):
    files = [x for x in os.listdir(target_root) if x.endswith(".txt")]
    for f in files:
        f_name = f.replace(".txt", "")
        origin_path = os.path.join(target_root, f)
        if f_name in valid:
            print(f_name)
            target_path = os.path.join(target_root, "valid", f)
        else:
            target_path = os.path.join(target_root, "train", f)
        shutil.move(origin_path, target_path)


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
    pattern1_num_dict = get_num_per_type(pattern1_annot_root)
    pattern2_num_dict = get_num_per_type(pattern2_annot_root)

    p1_train_annot, p1_valid_annot = split_valid(pattern1_annot_root, pattern1_num_dict)
    p2_train_annot, p2_valid_annot = split_valid(pattern2_annot_root, pattern2_num_dict)

    p1_root = os.path.join(root, "for_yolo_train", "pattern_1", "labels")
    mv_train_valid(p1_root, p1_train_annot, p1_valid_annot)

    p2_root = os.path.join(root, "for_yolo_train", "pattern_2", "labels")
    mv_train_valid(p2_root, p2_train_annot, p2_valid_annot)