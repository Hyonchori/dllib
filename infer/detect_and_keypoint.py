
import argparse
import os
import cv2
import numpy as np

import torch

from dllib.models.detector import BuildDetector
from dllib.data.for_infer.from_streams import LoadStreams
from dllib.data.for_infer.from_images import LoadImages, img_formats
from dllib.data.for_infer.from_videos import LoadVideo, vid_formats
from dllib.utils.bbox_utils import non_maximum_suppression
from dllib.utils.img_utils import colors, plot_one_box, plot_one_keypoint, letterbox



@torch.no_grad()
def main(opt):
    det_model = BuildDetector(backbone_cfg=opt.det_backbone_cfg,
                              neck_cfg=opt.det_neck_cfg,
                              detector_head_cfg=opt.det_head_cfg,
                              info=True,
                              head_mode="separate").cuda().eval()
    det_stride = det_model.stride
    if opt.det_weights is not None:
        if os.path.isfile(opt.det_weights):
            wts = torch.load(opt.det_weights)
            det_model.load_state_dict(wts)

    kp_model = BuildDetector(backbone_cfg=opt.kp_backbone_cfg,
                             neck_cfg=opt.kp_neck_cfg,
                             detector_head_cfg=opt.kp_head_cfg,
                             info=True,
                             head_mode="keypoint").cuda().eval()
    if opt.kp_weights is not None:
        if os.path.isfile(opt.kp_weights):
            wts = torch.load(opt.kp_weights)
            kp_model.load_state_dict(wts)
    device = next(det_model.parameters()).device

    webcam = opt.source.isnumeric() or opt.source.endswith(".txt") or opt.source.lower().startswith(
        ("rtsp://", "rtmp://", "http://", "https://")
    )
    if webcam:
        dataset = LoadStreams(opt.source, opt.det_img_size, det_stride, auto=opt.det_auto)
    else:
        files = os.listdir(opt.source)
        img_files = [x for x in files if x.split(".")[-1] in img_formats]
        vid_files = [x for x in files if x.split(".")[-1] in vid_formats]
        if img_files:
            dataset = LoadImages(opt.source, opt.det_img_size, det_stride, auto=opt.det_auto)
        elif vid_files:
            dataset = LoadVideo(opt.source, opt.det_img_size, det_stride, auto=opt.det_auto)
        else:
            raise Exception(f"No available dataset format for given source {opt.source}")

    if device.type != "cpu":
        sample = torch.zeros(1, 3, opt.det_img_size, opt.det_img_size).to(device).type_as(next(det_model.parameters()))
        det_model(sample)

        sample = torch.zeros(1, 3, opt.kp_img_size, opt.kp_img_size).to(device).type_as(next(kp_model.parameters()))
        kp_model(sample)

    det_labels = None
    if opt.det_labels is not None:
        with open(opt.det_labels) as f:
            data = f.readlines()
            det_labels = {i - 1: data[i - 1][: -1] for i in range(1, len(data) + 1)}

    kp_labels = None
    if opt.kp_labels is not None:
        with open(opt.kp_labels) as f:
            data = f.readlines()
            kp_labels = [data[i - 1][: -1] for i in range(1, len(data) + 1)]

    for img0, img, path in dataset:
        img0 = img0 if webcam else [img0]
        img_in = img.to(device).float() if not opt.half else img.to(device).half()
        img_in /= 255.

        det_pred = det_model(img_in)
        for i in range(len(det_pred)):
            bs, _, _, _, info = det_pred[i].shape
            det_pred[i] = det_pred[i].view(bs, -1, info)
        det_pred = torch.cat(det_pred, 1)
        det_pred = non_maximum_suppression(det_pred,
                                           conf_thr=opt.det_conf_thr,
                                           iou_thr=opt.det_iou_thr,
                                           target_cls=opt.det_target_cls)

        for i, (det) in enumerate(det_pred):
            im0 = np.ascontiguousarray(img[i].numpy().transpose(1, 2, 0))
            #cv2.imshow("ref", im0)
            im0_w, im0_h, _ = img0[i].shape
            im_w, im_h, _ = im0.shape
            r = max(im0_w / im_w, im0_h / im_h)

            kp_input_imgs = []
            bxs, bys, dws, dhs, ratios=  [], [], [], [], []
            if len(det):
                for *xyxy, conf, cls in reversed(det):
                    tmp_img = im0[max(0, int(xyxy[1])): int(xyxy[3]), max(0, int(xyxy[0])): int(xyxy[2]), :]
                    tmp_img, ratio, (dw, dh) = letterbox(tmp_img, opt.kp_img_size, auto=False)
                    kp_input_imgs.append(torch.from_numpy(tmp_img[np.newaxis, ...]))
                    bxs.append(torch.ones(1, 17, device=device) * xyxy[0])
                    bys.append(torch.ones(1, 17, device=device) * xyxy[1])
                    dws.append(torch.ones(1, 17, device=device) * dw)
                    dhs.append(torch.ones(1, 17, device=device) * dh)
                    ratios.append(torch.ones(1, 17, device=device) * ratio[0])

                    c = int(cls)
                    label = "{} {:.2f}".format(det_labels[c], conf) if det_labels is not None else None
                    plot_one_box(xyxy, im0, colors(c, True), label,) # scale=r)

            if kp_input_imgs:
                kp_input_imgs = torch.cat(kp_input_imgs).permute(0, 3, 1, 2).to(device) / 255.
                kp_input_imgs = kp_input_imgs.float() if not opt.half else kp_input_imgs.half()
                bxs = torch.cat(bxs)
                bys = torch.cat(bys)
                dws = torch.cat(dws)
                dhs = torch.cat(dhs)
                ratios = torch.cat(ratios)

                kp_pred = kp_model(kp_input_imgs)
                kp_pred[..., 0::3] = ((kp_pred[..., 0::3] * opt.kp_img_size - dws) / ratios + bxs) #* r
                kp_pred[..., 1::3] = ((kp_pred[..., 1::3] * opt.kp_img_size - dhs) / ratios + bys) #* r
                for kp in kp_pred:
                    plot_one_keypoint(kp, im0)

            cv2.imshow("img0", img0[i])
            cv2.imshow(f"im{i}", im0)
            cv2.waitKey(1)


def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument("--det_backbone_cfg", type=str, help="detection backbone.yaml path",
                        default="../models/cfgs/base_backbone_m.yaml")
    parser.add_argument("--det_neck_cfg", type=str, help="detection neck.yaml path",
                        default="../models/cfgs/base_neck_m.yaml")
    parser.add_argument("--det_head_cfg", type=str, help="detection head.yaml path",
                        default="../models/cfgs/base_detection_head2.yaml")
    parser.add_argument("--det_weights", type=str, help="weights of detection model",
                        default="../weights/base_detector.pt")
    parser.add_argument("--det_img_size", type=int, default=416)
    parser.add_argument("--det_auto", type=bool, default=True)
    parser.add_argument("--det_labels", type=str, default="../data/for_train/coco_labels91.txt")
    parser.add_argument("--det_conf_thr", type=float, default=0.0001)
    parser.add_argument("--det_iou_thr", type=float, default=0.05)
    parser.add_argument("--det_target_cls", type=int, default=None)

    parser.add_argument("--kp_backbone_cfg", type=str, help="keypoint backbone.yaml path",
                        default="../models/cfgs/keypoint_backbone.yaml")
    parser.add_argument("--kp_neck_cfg", type=str, help="keypoint neck.yaml path",
                        default=None)
    parser.add_argument("--kp_head_cfg", type=str, help="keypoint head.yaml path",
                        default="../models/cfgs/keypoint_detection_head.yaml")
    parser.add_argument("--kp_weights", type=str, help="weights of keypoint model",
                        default="../weights/keypoint_detector.pt")
    parser.add_argument("--kp_img_size", type=int, default=128)
    parser.add_argument("--kp_v_thr", type=float, default=0.7)
    parser.add_argument("--kp_labels", type=str, default="../data/for_train/coco_keypoint_labels.txt")

    source = "rtsp://datonai:datonai@172.30.1.49:554/stream1"
    parser.add_argument("--source", type=str, default=source)
    parser.add_argument("--half", action="store_true", help="use FP16 half-precision inference")
    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt


if __name__ == "__main__":
    opt = parse_opt(True)
    main(opt)
