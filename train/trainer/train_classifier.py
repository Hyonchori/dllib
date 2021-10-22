
import os
import argparse
import copy
import time
import csv

import torch
import torch.nn as nn
import torch.optim as optim
import albumentations as A
from tqdm import tqdm

from dllib.models.classifier import BuildClassifier
from dllib.data.for_train.military_civil_dataset import get_mc_train_dataloader, \
    get_mc_valid_dataloader
from dllib.train.lr_schedulers import CosineAnnealingWarmUpRestarts
from dllib.train.losses import ComputeClfLoss, FocalLoss


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
    device = next(model.parameters()).device

    train_transform = A.Compose([
        A.RandomBrightnessContrast(),
        A.Cutout(max_h_size=10, max_w_size=10)
    ])
    train_dataloader = get_mc_train_dataloader(img_size=opt.img_size,
                                               batch_size=opt.batch_size,
                                               transform=train_transform)
    valid_dataloader = get_mc_valid_dataloader(img_size=opt.img_size,
                                               batch_size=opt.batch_size,
                                               transform=None)

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.00001, weight_decay=0.003, momentum=0.9)
    #optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.003)
    lr_sch = CosineAnnealingWarmUpRestarts(optimizer, T_0=10, T_mult=2, eta_max=0.0001, T_up=3, gamma=0.95)

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
        print(f"\n--- Epoch: {e}")
        time.sleep(0.5)
        train_loss = train(model, optimizer, e, train_dataloader, loss_fn, device)
        time.sleep(0.5)
        print(train_loss)

        valid_loss = evaluate(model, e, valid_dataloader, loss_fn, device)
        time.sleep(0.5)
        print(valid_loss)

        if os.path.isfile(log_save_dir):
            with open(log_save_dir, "r") as f:
                reader = csv.reader(f)
                logs = list(reader)
                logs.append([train_loss] + [valid_loss] + [optimizer.param_groups[0]["lr"]])
            with open(log_save_dir, "w") as f:
                writer = csv.writer(f)
                for log in logs:
                    writer.writerow(log)
        else:
            with open(log_save_dir, "w") as f:
                writer = csv.writer(f)
                writer.writerow([train_loss] + [valid_loss] + [optimizer.param_groups[0]["lr"]])

        if valid_loss < best_loss:
            best_loss = valid_loss
            best_model_wts = copy.deepcopy(model.state_dict())

        #lr_sch.step()
        if e % save_interval == 0:
            torch.save(best_model_wts, wts_save_dir)


def train(model, optimizer, epoch, dataloader, loss_fn, device):
    model.train()
    train_loss = 0.
    nb = len(dataloader)
    pbar = enumerate(dataloader)
    pbar = tqdm(pbar, total=nb)
    for i, (img0, img_b, img_cls, img_name) in pbar:
        img_b = img_b.to(device) / 255.
        img_cls = torch.from_numpy(img_cls).to(device).float()

        optimizer.zero_grad()
        pred = model(img_b.float(), epoch)[0]
        loss = loss_fn(pred, img_cls)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(pbar)
    return train_loss


@torch.no_grad()
def evaluate(model, epoch, dataloader, loss_fn, device):
    model.train()
    train_loss = 0.
    nb = len(dataloader)
    pbar = enumerate(dataloader)
    pbar = tqdm(pbar, total=nb)
    for i, (img0, img_b, img_cls, img_name) in pbar:
        img_b = img_b.to(device) / 255.
        img_cls = torch.from_numpy(img_cls).to(device).float()

        pred = model(img_b.float(), epoch)[0]
        loss = loss_fn(pred, img_cls)

        train_loss += loss.item()

    train_loss /= len(pbar)
    return train_loss


def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone_cfg", type=str,
                        default="../../models/cfgs/backbone_clf.yaml")
    parser.add_argument("--neck_cfg", type=str,
                        default=None)
    parser.add_argument("--head_cfg", type=str,
                        default="../../models/cfgs/base_classification_head.yaml")
    parser.add_argument("--weights", type=str,
                        default="../../weights/military_civil_clf.pt")
    parser.add_argument("--start_epoch", type=int, default=0)
    parser.add_argument("--end_epoch", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--img_size", type=int, default=128)
    parser.add_argument("--save_dir", type=str, default="../../weights")
    parser.add_argument("--name", type=str, default="military_civil_clf3")
    parser.add_argument("--save_interval", type=int, default=10)
    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt


if __name__ == "__main__":
    opt = parse_opt(True)
    main(opt)
