
import os
import argparse
import cv2
import numpy as np
import time

import torch

from dllib.data.for_train.coco_dataset import get_coco2017dataloader, get_coco2017_valid_dataloader
from dllib.models.detector import BuildDetector
from dllib.utils.img_utils import plot_one_keypoint


@torch.no_grad()
def main(opt):
    model = BuildDetector(backbone_cfg=opt.backbone_cfg,
                          neck_cfg=opt.neck_cfg,
                          detector_head_cfg=opt.head_cfg,
                          info=True,
                          head_mode="keypoint").cuda()
    if opt.weights is not None:
        if os.path.isfile(opt.weights):
            wts = torch.load(opt.weights)
            model.load_state_dict(wts)
    device = next(model.parameters()).device

    train_dataloader, valid_dataloader = get_coco2017dataloader(img_size=opt.img_size,
                                                                mode="only keypoint",
                                                                train_batch=opt.batch_size,
                                                                valid_batch=opt.batch_size,)
    save_dir = opt.save_dir
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    model_name = opt.name
    v_thr = opt.v_thr
    labels = None
    if opt.labels is not None:
        with open(opt.labels) as f:
            data = f.readlines()
            labels = np.array([data[i - 1][: -1] for i in range(1, len(data) + 1)])
    print(labels)

    model.eval()
    for img_b, keypoint_b, img_name in valid_dataloader:
        img0 = img_b.numpy().transpose(0, 2, 3, 1)
        img_b = img_b.to(device) / 255.
        pred = model(img_b.float())

        for im0, p, keypoint in zip(img0, pred, keypoint_b):
            print("\n---")
            comp_img = im0.copy()
            img = im0.copy()
            pred_kp = p.cpu().detach().numpy()
            xs = pred_kp[..., 0::3] * opt.img_size
            ys = pred_kp[..., 1::3] * opt.img_size
            vs = pred_kp[..., 2::3]
            valid_kp = vs > v_thr
            valid_kp_list = labels[valid_kp]
            print(valid_kp_list)
            pred_kp = []
            for x, y, v in zip(xs, ys, vs):
                pred_kp += [x]
                pred_kp += [y]
                pred_kp += [v]

            plot_one_keypoint(keypoint, img, v_thr=v_thr)
            plot_one_keypoint(pred_kp, comp_img, v_thr=v_thr)
            cv2.imshow("img", img)
            cv2.imshow("comp_img", comp_img)
            cv2.waitKey(0)




def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone_cfg", type=str, help="backbone.yaml path",
                        default="../models/cfgs/keypoint_backbone.yaml")
    parser.add_argument("--neck_cfg", type=str, help="backbone.yaml path",
                        default=None)
    parser.add_argument("--head_cfg", type=str, help="backbone.yaml path",
                        default="../models/cfgs/keypoint_detection_head.yaml")
    parser.add_argument("--weights", type=str, help="initial weights path",
                        default="../weights/keypoint_detector.pt")
    parser.add_argument("--start_epoch", type=int, default=0)
    parser.add_argument("--end_epoch", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--img_size", type=int, default=128)
    parser.add_argument("--save_dir", type=str, default="../weights")
    parser.add_argument("--name", type=str, default="keypoint_detector")
    parser.add_argument("--save_interval", type=int, default=10)

    parser.add_argument("--v_thr", type=float, default=0.3)
    parser.add_argument("--labels", type=str, default="../data/for_train/coco_keypoint_labels.txt")
    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt


if __name__ == "__main__":
    opt = parse_opt(True)
    main(opt)

