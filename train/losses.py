
import torch
import torch.nn as nn

from dllib.utils.metrics import bbox_iou
from dllib.utils.bbox_utils import cpwh2xyxy, xywh2xyxy


class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class ComputeDetectionLoss:
    def __init__(self, model, iou_thr=0.4):
        self.device = next(model.parameters()).device

        self.BCEcls = FocalLoss(nn.BCEWithLogitsLoss())
        self.BCEobj = FocalLoss(nn.BCEWithLogitsLoss())
        self.iou_thr = iou_thr
        self.nc = model.head.nc

    def __call__(self, pred, targets, epoch=None):
        # pred: (bs, total_predicted_bbox_num, 4 + 1 + cls_num)
        # target: (bs, total_target_bbox_num, 4 + 1 + cls)
        total_loss = torch.zeros(3, device=self.device)
        for p, t in zip(pred, targets):
            p_xyxys = cpwh2xyxy(p)
            t_xyxys = xywh2xyxy(t)

            loss_per_img = torch.zeros(3, device=self.device)
            n = 1
            for t_xyxy in t_xyxys:
                ious = bbox_iou(p_xyxys, t_xyxy)
                valid_idx = ious > self.iou_thr
                valid_p = p_xyxys[valid_idx]
                if valid_p.shape[0] == 0:
                    continue
                n += 1
                zero_idx = ious == 0
                zero_p = p_xyxys[zero_idx]
                if zero_p.shape[0] != 0:
                    target_pos_conf = torch.ones((valid_p.shape[0], 1), device=self.device)
                    target_neg_conf = torch.zeros((zero_p.shape[0], 1), device=self.device)
                    conf_loss_pos = self.BCEobj(valid_p[:, 4:5], target_pos_conf)
                    conf_loss = torch.mean(conf_loss_pos + self.BCEobj(zero_p[:, 4:5], target_neg_conf) * 0.3)
                else:
                    conf_loss = torch.tensor(0, device=self.device)


                iou_loss = torch.mean(1 - ious[valid_idx])

                target_cls = int(t_xyxy[-1].item())
                target_cls_onehot = torch.zeros((valid_p.shape[0], self.nc), device=self.device)
                target_cls_onehot[:, target_cls - 1] = 1.
                pred_cls = valid_p[:, 5:]
                cls_loss = self.BCEcls(pred_cls, target_cls_onehot)

                loss_per_img[0] += iou_loss
                loss_per_img[1] += conf_loss
                loss_per_img[2] += cls_loss

                #loss_per_img += loss
            loss_per_img /= n
            total_loss += loss_per_img
        total_loss /= len(pred)
        return total_loss
