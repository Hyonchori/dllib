
import cv2
import torch
import numpy as np

from dllib.utils.bbox_utils import xywh2xyxy


class Colors:
    # Ultralytics color palette https://ultralytics.com/
    def __init__(self):
        # hex = matplotlib.colors.TABLEAU_COLORS.values()
        hex = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
               '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')
        self.palette = [self.hex2rgb('#' + c) for c in hex]
        self.n = len(self.palette)

    def __call__(self, i, bgr=False, norm=False):
        c = self.palette[int(i) % self.n] if not norm else [x/255. for x in self.palette[int(i) % self.n]]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):  # rgb order (PIL)
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))


colors = Colors()


def letterbox(img,
              new_shape=(640, 480),
              color=(114, 114, 114),
              auto=True,
              scaleFill=False,
              scaleup=True,
              stride=32):
    shape = img.shape[: 2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)

    ratio = r, r
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    if auto:
        dw = stride - np.mod(new_unpad[0], stride) if np.mod(new_unpad[0], stride) != 0 else 0
        dh = stride - np.mod(new_unpad[1], stride) if np.mod(new_unpad[1], stride) != 0 else 0
    elif scaleFill:
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]

    dw /= 2
    dh /= 2

    if shape[::-1] != new_shape:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, ratio, (dw, dh)


def plot_one_box(x: torch.tensor,
                 im: np.ndarray,
                 color: tuple=(128, 128, 128),
                 label: str=None,
                 line_thickness: int=2,
                 scale: int=1.):
    # Plots one bounding box on image 'im' using OpenCV
    assert im.data.contiguous, 'Image not contiguous. Apply np.ascontiguousarray(im) to plot_on_box() input image.'
    tl = line_thickness or round(0.002 * (im.shape[0] + im.shape[1]) / 2) + 1  # line/font thickness
    c1, c2 = (int(x[0] * scale), int(x[1] * scale)), (int(x[2] * scale), int(x[3] * scale))
    cv2.rectangle(im, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(im, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(im, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


def plot_one_keypoint(kp: torch.tensor,
                      im: np.ndarray,
                      color: tuple=(255, 255, 255),
                      radius: int=2,
                      v_thr: int=0.2):
    xs = kp[0::3]
    ys = kp[1::3]
    vs = kp[2::3]

    for x, y, v in zip(xs, ys, vs):
        if v > v_thr:
            cv2.circle(im, (int(x), int(y)), radius, color, -1)


def crop_bbox(img: np.ndarray,
              bboxes: list,
              keypoints: list,
              letterbox_size: (int, int)=None,
              normalize: bool=True):
    if isinstance(letterbox_size, int):
        letterbox_size = (letterbox_size, letterbox_size)

    imgs = []
    adj_keypoints = []
    for bbox, keypoint in zip(bboxes, keypoints):
        if sum(keypoint) == 0:
            continue

        xs = keypoint[0::3]
        ys = keypoint[1::3]
        vs = keypoint[2::3]
        adj_keypoint = []

        bbox = list(map(int, bbox))
        cropped_img = img[bbox[1]: bbox[3], bbox[0]: bbox[2]]
        if letterbox_size is not None:
            cropped_img, ratio, (dw, dh) = letterbox(cropped_img, letterbox_size, auto=False)
            norm = (1, 1) if not normalize else cropped_img.shape[:2]
            for x, y, v in zip(xs, ys, vs):
                adj_keypoint += [((x - bbox[0]) * ratio[0] + dw) / norm[0]] \
                    if x != 0 else [0]
                adj_keypoint += [((y - bbox[1]) * ratio[1] + dh) / norm[1]] \
                    if y != 0 else [0]
                adj_keypoint += [1 if v > 0 else 0]
        else:
            norm = (1, 1) if not normalize else cropped_img.shape[:2]
            for x, y, v in zip(xs, ys, vs):
                adj_keypoint += [(x - bbox[0]) / norm[0]] if x != 0 else [0]
                adj_keypoint += [(y - bbox[1]) / norm[1]] if y != 0 else [0]
                adj_keypoint += [1 if v > 0 else 0]
        imgs.append(cropped_img)
        adj_keypoints.append(adj_keypoint)
    return np.array(imgs), np.array(adj_keypoints)