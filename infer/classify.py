
import os
import argparse
import copy
import time
import csv
import sys

import torch
import cv2

from dllib.models.classifier import BuildClassifier
from dllib.data.for_train.military_civil_dataset import get_mc_train_dataloader, \
    get_mc_valid_dataloader


@torch.no_grad()
def main(opt):
    model = BuildClassifier(backbone_cfg=opt.backbone_cfg,
                            neck_cfg=opt.neck_cfg,
                            head_cfg=opt.head_cfg,
                            info=True).cuda()
    if opt.weights is not None:
        if os.path.isfile(opt.weights):
            wts = torch.load(opt.weights)
            model.load_state_dict(wts)
            print("Model is initialized with existing weights!")
        else:
            print("Model is initialized!")
    torch.save(model, "military_civil_clf_v1.pt")
    sys.exit()
    
    device = next(model.parameters()).device

    train_dataloader = get_mc_train_dataloader(img_size=opt.img_size,
                                               batch_size=opt.batch_size,
                                               transform=None)
    valid_dataloader = get_mc_valid_dataloader(img_size=opt.img_size,
                                               batch_size=opt.batch_size,
                                               transform=None)

    for img, img_tensor, label, path in train_dataloader:
        img_tensor = img_tensor.to(device) / 255.
        pred = model(img_tensor.float())[0]
        print(pred)
        pred_label = torch.argmax(pred, dim=1)
        for i, im in enumerate(img):
            cv2.imshow(f"{train_dataloader.dataset.clss[pred_label[i]]} {pred[i][pred_label[i]]}", im[:, :, ::-1])
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        break


def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone_cfg", type=str,
                        default="../models/cfgs/backbone_clf.yaml")
    parser.add_argument("--neck_cfg", type=str,
                        default=None)
    parser.add_argument("--head_cfg", type=str,
                        default="../models/cfgs/base_classification_head.yaml")
    parser.add_argument("--weights", type=str,
                        default="../weights/military_civil_clf3.pt")
    parser.add_argument("--start_epoch", type=int, default=0)
    parser.add_argument("--end_epoch", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--img_size", type=int, default=128)
    parser.add_argument("--save_dir", type=str, default="../weights")
    parser.add_argument("--name", type=str, default="military_civil_clf")
    parser.add_argument("--save_interval", type=int, default=10)
    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt


if __name__ == "__main__":
    opt = parse_opt(True)
    main(opt)
