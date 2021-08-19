
import argparse
import os
import time
import cv2
import numpy as np

import torch
import torchvision
import torch.backends.cudnn as cudnn

from dllib.models.detector import BuildDetector
from dllib.data.for_infer.from_streams import LoadStreams
from dllib.utils.bbox_utils import non_maximum_suppression
from dllib.utils.img_utils import colors, plot_one_box


@torch.no_grad()
def main(opt):
    model = BuildDetector(backbone_cfg=opt.backbone_cfg,
                          neck_cfg=opt.neck_cfg,
                          detector_head_cfg=opt.head_cfg,
                          info=True).cuda().eval()
    stride = model.stride
    if opt.weights is not None:
        if os.path.isfile(opt.weights):
            wts = torch.load(opt.weights)
            model.load_state_dict(wts)
    device = next(model.parameters()).device

    webcam = opt.source.isnumeric() or opt.source.endswith(".txt") or opt.source.lower().startswith(
        ("rtsp://", "rtmp://", "http://", "https://")
    )
    if webcam:
        cudnn.benchmark = True
        dataset = LoadStreams(opt.source, opt.img_size, stride)
        bs = len(dataset)
    else:
        pass

    if device.type != "cpu":
        sample = torch.zeros(1, 3, opt.img_size, opt.img_size).to(device).type_as(next(model.parameters()))
        model(sample)

    labels = None
    if opt.labels is not None:
        with open(opt.labels) as f:
            data = f.readlines()
            labels = {i: data[i - 1][:-1] for i in range(1, len(data) + 1)}

    for img0, img, path in dataset:
        img_in = img.to(device).float() / 255.

        pred = model(img_in)
        for i in range(len(pred)):
            bs, _, _, _, info = pred[i].shape
            pred[i] = pred[i].view(bs, -1, info)
        pred = torch.cat(pred, 1)

        pred = non_maximum_suppression(pred, conf_thr=opt.conf_thr, iou_thr=opt.iou_thr, target_cls=opt.target_cls)
        for i, (det) in enumerate(pred):
            im0 = np.ascontiguousarray(img[i].numpy().transpose(1, 2, 0))
            if len(det):
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls) + 1
                    label = labels[c] if labels is not None else None
                    plot_one_box(xyxy, im0, c, label)

            cv2.imshow("img0", img0[i])
            cv2.imshow(f'im{i}', im0)
            cv2.waitKey(1)




def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone_cfg", type=str, help="backbone.yaml path",
                        default="../models/cfgs/base_backbone_m.yaml")
    parser.add_argument("--neck_cfg", type=str, help="backbone.yaml path",
                        default="../models/cfgs/base_neck_m.yaml")
    parser.add_argument("--head_cfg", type=str, help="backbone.yaml path",
                        default="../models/cfgs/base_detection_head_m.yaml")
    parser.add_argument("--weights", type=str, help="initial weights path")
    parser.add_argument("--img_size", type=int, default=412)
    parser.add_argument("--source", type=str, default="0")
    parser.add_argument("--labels", type=str, default="../data/for_train/coco_labels91.txt")
    parser.add_argument("--target_cls", type=int, default=0)
    parser.add_argument("--conf_thr", type=float, default=0.25)
    parser.add_argument("--iou_thr", type=float, default=0.4)
    parser.add_argument("--save_dir", type=str, default="../runs")
    parser.add_argument("--name", type=str, default="base_detector")
    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt




if __name__ == "__main__":
    opt = parse_opt(True)
    main(opt)