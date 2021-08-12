
import torch
import torchvision
import time
import numpy as np

from dllib.utils.metrics import box_iou


def xyxy2cpwh(x):  # (x, y) is center point
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y


def cpwh2xyxy(x):  # (x, y) is center point
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def xywh2xyxy(x):  # (x, y) is top-left point
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 2] = x[:, 0] + x[:, 2]  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3]  # bottom right y
    return y


def letterboxed_xywh(x, ratio, dw, dh):
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] * ratio[0] + dw
    y[:, 1] = x[:, 1] * ratio[1] + dh
    y[:, 2] = x[:, 2] * ratio[0]
    y[:, 3] = x[:, 3] * ratio[1]
    return y


def letterboxed_keypoint(x, ratio, dw, dh):
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0::3] = x[:, 0::3] * ratio[0] + dw
    y[:, 1::3] = x[:, 1::3] * ratio[1] + dh
    return y


def normalize_xyxy(x, w, h):
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] / w
    y[:, 1] = x[:, 1] / h
    y[:, 2] = x[:, 2] / w
    y[:, 3] = x[:, 3] / h
    return y


def unnormalize_xyxy(x, w, h):
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] * w
    y[:, 1] = x[:, 1] * h
    y[:, 2] = x[:, 2] * w
    y[:, 3] = x[:, 3] * h
    return y


def normalize_keypoint(x, w, h):
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0::3] = x[:, 0::3] / w
    y[:, 1::3] = x[:, 1::3] / h
    return y


def unnormalize_keypoint(x, w, h):
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0::3] = x[:, 0::3] * w
    y[:, 1::3] = x[:, 1::3] * h
    return y


def non_maximum_suppression(pred, conf_thr=0.4, iou_thr=0.45, target_cls=None, multi_label=False,
                            agnostic=False, labels=(), max_det=300):
    # Checks
    assert 0 <= conf_thr <= 1, f'Invalid Confidence threshold {conf_thr}, valid values are between 0.0 and 1.0'
    assert 0 <= iou_thr <= 1, f'Invalid IoU {iou_thr}, valid values are between 0.0 and 1.0'

    print(pred.shape)
    nc = pred.shape[2] - 5
    xc = pred[..., 4] > conf_thr

    min_wh, max_wh = 2, 4096
    max_nms = 30000
    time_limit = 10.0
    redundant = True
    multi_label &= nc > 1
    merge = False

    t = time.time()
    output = [torch.zeros((0, 6), device=pred.device)] * pred.shape[0]
    for xi, x in enumerate(pred):
        x = x[xc[xi]]

        if labels and len(labels[xi]):
            l = labels[xi]
            v = torch.zeros((len(l), nc + 5), device=x.device)
            v[:, :4] = l[:, 1:5]
            v[:, 4] = 1.0
            v[range(len(l)), l[:, 0].long() + 5] = 1.0
            x = torch.cat((x, v), 0)

        if not x.shape[0]:
            continue

        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        box = cpwh2xyxy(x[:, :4])

        if multi_label:
            i, j = (x[:, 5:] > conf_thr).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thr]

        if target_cls is not None:
            x = x[(x[:, 5:6] == torch.tensor(target_cls, device=x.device)).any(1)]

        n = x.shape[0]
        if not n:
            continue
        elif n > max_nms:
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]

        c = x[:, 5:6] * (0 if agnostic else max_wh)
        boxes, scores = x[:, :4] + c, x[:, 4]
        i = torchvision.ops.nms(boxes, scores, iou_thr)
        if i.shape[0] > max_det:
            i = i[:max_det]
        if merge and (1 < n < 3E3):
            iou = box_iou(boxes[i], boxes) > iou_thr
            weights = iou * scores[None]
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)
            if redundant:
                i = i[iou,sum(1) > 1]

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            print(f"WARNING: NMS time limit {time_limit}s exceeded")
            break
    return output
