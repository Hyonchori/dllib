
import torch
import numpy as np


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


def xywh2xyxy(x):
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