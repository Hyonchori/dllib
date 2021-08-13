
import torch

from dllib.train.losses import detection_loss


if __name__ == "__main__":
    from dllib.data.for_train.coco_dataset import get_coco2017_valid_dataloader
    from dllib.models.detector import BuildDetector
    model = BuildDetector(backbone_cfg="../../models/cfgs/base_backbone_m.yaml",
                          neck_cfg="../../models/cfgs/base_neck_m.yaml",
                          detector_head_cfg="../../models/cfgs/base_detection_head_m.yaml",
                          info=True)

    dataloader = get_coco2017_valid_dataloader(img_size=(412, 412),
                                               mode="detection",
                                               batch_size=8)
    for img0, img_b, bboxes0, bbox_b, img_name in dataloader:
        pred = model(img_b.float())
        for i in range(len(pred)):
            bs, _, _, _, info = pred[i].shape
            pred[i] = pred[i].view(bs, -1, info)
        pred = torch.cat(pred, 1)

        loss = detection_loss(pred, bbox_b)

        break