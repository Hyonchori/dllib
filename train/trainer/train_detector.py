
import torch
import torch.optim as optim
from tqdm import tqdm

from dllib.train.losses import ComputeDetectionLoss


if __name__ == "__main__":
    from dllib.data.for_train.coco_dataset import get_coco2017_valid_dataloader, get_coco2017_train_dataloader
    from dllib.models.detector import BuildDetector

    model = BuildDetector(backbone_cfg="../../models/cfgs/base_backbone_m.yaml",
                          neck_cfg="../../models/cfgs/base_neck_m.yaml",
                          detector_head_cfg="../../models/cfgs/base_detection_head_m.yaml",
                          info=True).cuda()
    device = next(model.parameters()).device

    dataloader = get_coco2017_train_dataloader(img_size=(412, 412),
                                               mode="detection",
                                               batch_size=16)

    compute_loss = ComputeDetectionLoss(model)
    opt = optim.Adam(model.parameters(), lr=0.0001)
    nb = len(dataloader)

    epoch = 5
    for e in range(epoch):
        print("\n--- {}".format(e))
        l = 0
        pbar = enumerate(dataloader)
        pbar = tqdm(pbar, total=nb)
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
        l /= len(dataloader)
        print(l)