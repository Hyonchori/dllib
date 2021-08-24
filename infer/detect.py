
import argparse
import os
import cv2
import numpy as np

import torch
import torch.backends.cudnn as cudnn

from dllib.models.detector import BuildDetector
from dllib.data.for_infer.from_streams import LoadStreams
from dllib.data.for_infer.from_images import LoadImages, img_formats
from dllib.data.for_infer.from_videos import LoadVideo, vid_formats
from dllib.utils.bbox_utils import non_maximum_suppression
from dllib.utils.img_utils import colors, plot_one_box


@torch.no_grad()
def main(opt):
    model = BuildDetector(backbone_cfg=opt.backbone_cfg,
                          neck_cfg=opt.neck_cfg,
                          detector_head_cfg=opt.head_cfg,
                          info=True).cuda().eval()
    if opt.half:
        model = model.half()
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
        dataset = LoadStreams(opt.source, opt.img_size, stride, auto=opt.auto)
    else:
        files = os.listdir(opt.source)
        img_files = [x for x in files if x.split(".")[-1] in img_formats]
        vid_files = [x for x in files if x.split(".")[-1] in vid_formats]
        if img_files:
            dataset = LoadImages(opt.source, opt.img_size, stride, auto=opt.auto)
        elif vid_files:
            dataset = LoadVideo(opt.source, opt.img_size, stride, auto=opt.auto)

    if device.type != "cpu":
        sample = torch.zeros(1, 3, opt.img_size, opt.img_size).to(device).type_as(next(model.parameters()))
        model(sample)

    labels = None
    if opt.labels is not None:
        with open(opt.labels) as f:
            data = f.readlines()
            labels = {i: data[i - 1][:-1] for i in range(1, len(data) + 1)}

    for img0, img, path in dataset:
        img0 = img0 if webcam else [img0]
        img_in = img.to(device).float() if not opt.half else img.to(device).half()
        img_in /= 255.

        pred = model(img_in)
        for i in range(len(pred)):
            bs, _, _, _, info = pred[i].shape
            pred[i] = pred[i].view(bs, -1, info)
        pred = torch.cat(pred, 1)

        pred = non_maximum_suppression(pred, conf_thr=opt.conf_thr, iou_thr=opt.iou_thr, target_cls=opt.target_cls)

        for i, (det) in enumerate(pred):
            im0 = np.ascontiguousarray(img[i].numpy().transpose(1, 2, 0))
            im0 = cv2.cvtColor(im0, cv2.COLOR_RGB2BGR)
            if len(det):
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls) + 1
                    label = "{} {:.2f}".format(labels[c], conf) if labels is not None else None
                    plot_one_box(xyxy, im0, colors(c, True), label)

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
                        default="../models/cfgs/base_detection_head2.yaml")
    parser.add_argument("--weights", type=str, help="initial weights path",
                        default="../weights/base_detetor.pt")
    parser.add_argument("--img_size", type=int, default=640)

    source = "/media/daton/D6A88B27A88B0569/dataset/mot/MOT17/test/MOT17-01-DPM/img1"
    source = "/home/daton/PycharmProjects/pythonProject/datonlib/videos"
    parser.add_argument("--source", type=str, default=source)
    parser.add_argument("--auto", type=bool, default=True)
    parser.add_argument("--labels", type=str, default="../data/for_train/coco_labels91.txt")
    parser.add_argument("--target_cls", type=int, default=None)
    parser.add_argument("--conf_thr", type=float, default=0.001)
    parser.add_argument("--iou_thr", type=float, default=0.05)
    parser.add_argument("--half", action="store_true", help='use FP16 half-precision inference')
    parser.add_argument("--save_dir", type=str, default="../runs")
    parser.add_argument("--name", type=str, default="base_detector")
    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt




if __name__ == "__main__":
    opt = parse_opt(True)
    main(opt)