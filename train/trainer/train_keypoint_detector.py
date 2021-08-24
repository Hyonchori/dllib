
import os
import argparse
import copy
import time
import csv
from tqdm import tqdm

import torch
import torch.optim as optim
import albumentations as A

from dllib.train.losses import ComputeDetectionLoss
from dllib.train.lr_schedulers import CosineAnnealingWarmUpRestarts
from dllib.data.for_train.coco_dataset import get_coco2017dataloader, get_coco2017_valid_dataloader
from dllib.models.detector import BuildDetector


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

    train_transform = A.Compose([
        A.ColorJitter(),
        A.HueSaturationValue(),
        A.RandomBrightnessContrast(),
    ])
    train_dataloader, valid_dataloader = get_coco2017dataloader(img_size=opt.img_size,
                                                                mode="only keypoint",
                                                                train_batch=opt.batch_size,
                                                                valid_batch=opt.batch_size,
                                                                train_transform=train_transform)
    save_dir = opt.save_dir
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    model_name = opt.name
    wts_save_dir = os.path.join(save_dir, model_name + ".pt")
    log_save_dir = os.path.join(save_dir, model_name + "_log.csv")
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 100.
    save_interval = opt.save_interval

    start_epoch = opt.start_epoch
    end_epoch = opt.end_epoch
    for e in range(start_epoch, end_epoch + 1):
        print("\n--- {}".format(e))
        time.sleep(0.5)
        for img, keypoint_b, img_name in train_dataloader:
            print(img.shape, keypoint_b.shape)
            break
        #train_loss = train(model, optimizer, e, train_dataloader, compute_loss, loss_weight, device)



def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone_cfg", type=str, help="backbone.yaml path",
                        default="../../models/cfgs/base_backbone_m.yaml")
    parser.add_argument("--neck_cfg", type=str, help="backbone.yaml path",
                        default=None)
    parser.add_argument("--head_cfg", type=str, help="backbone.yaml path",
                        default="../../models/cfgs/keypoint_detection_head.yaml")
    parser.add_argument("--weights", type=str, help="initial weights path",
                        default="../../weights/keypoint_detector.pt")
    parser.add_argument("--start_epoch", type=int, default=0)
    parser.add_argument("--end_epoch", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--img_size", type=int, default=128)
    parser.add_argument("--save_dir", type=str, default="../../weights")
    parser.add_argument("--name", type=str, default="keypoint_detector")
    parser.add_argument("--save_interval", type=int, default=10)
    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt


if __name__ == "__main__":
    opt = parse_opt(True)
    main(opt)

