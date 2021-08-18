
import argparse
import torch
import torch.optim as optim
from tqdm import tqdm
import time

from dllib.train.losses import ComputeDetectionLoss


def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="model.yaml path")
    parser.add_argument("--weights", type=str, help="initial weights path")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--image_size", type=int, default=412)
    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt


def main(opt):
    pass


if __name__ == "__main__":
    from dllib.data.for_train.coco_dataset import get_coco2017_valid_dataloader, get_coco2017_train_dataloader
    from dllib.models.detector import BuildDetector

    model = BuildDetector(backbone_cfg="../../models/cfgs/base_backbone_m.yaml",
                          neck_cfg="../../models/cfgs/base_neck_m.yaml",
                          detector_head_cfg="../../models/cfgs/base_detection_head_m.yaml",
                          info=True).cuda()
    device = next(model.parameters()).device

    dataloader = get_coco2017_valid_dataloader(img_size=(412, 412),
                                               mode="detection",
                                               batch_size=18)

    compute_loss = ComputeDetectionLoss(model)
    opt = optim.Adam(model.parameters(), lr=0.0001)
    nb = len(dataloader)

    epoch = 5
    for e in range(epoch):
        print("\n--- {}".format(e))
        l = 0
        pbar = enumerate(dataloader)
        pbar = tqdm(pbar, total=nb)
        time.sleep(0.5)
        for i, (img0, img_b, bboxes0, bbox_b, img_name) in pbar:
            img_b = img_b.to(device)
            for i, bbox in enumerate(bbox_b):
                bbox_b[i] = bbox_b[i].to(device)
            opt.zero_grad()

            pred = model(img_b.float())

            for i in range(len(pred)):
                bs, _, _, _, no = pred[i].shape
                pred[i] = pred[i].view(bs, -1, no)
            pred = torch.cat(pred, dim=1)

            loss = compute_loss(pred, bbox_b)
            loss.backward()
            opt.step()
            l += loss.item()
        l /= float(len(dataloader))
        print(l)