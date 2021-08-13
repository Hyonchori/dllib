
import torch
import torch.nn as nn

from dllib.models.builder.backbone import BuildBackbone
from dllib.models.builder.neck import BuildNeck
from dllib.models.builder.detection_head import BuildDetectionHead
from dllib.utils.model_utils import model_info


class BuildDetector(nn.Module):
    def __init__(self,
                 backbone_cfg,
                 neck_cfg,
                 detector_head_cfg,
                 info=False):
        super().__init__()
        self.backbone = BuildBackbone(backbone_cfg)
        self.neck = BuildNeck(neck_cfg)
        self.head = BuildDetectionHead(detector_head_cfg)

        self.mode = "detector"
        self.input_shape = self.backbone.input_shape
        if info:
            self.info(verbose=False)

    def info(self, verbose=False, batch_size=1):
        model_info(self, verbose, self.input_shape, batch_size)

    def forward(self, x, epoch=None):
        x = self.backbone(x, epoch)
        x = self.neck(x, epoch)
        x = self.head(x, epoch)
        return x


if __name__ == "__main__":
    bb_cfg = "cfgs/base_backbone_m.yaml"
    n_cfg = "cfgs/base_neck_m.yaml"
    h_cfg = "cfgs/base_detection_head_m.yaml"
    model = BuildDetector(bb_cfg,
                          n_cfg,
                          h_cfg,
                          True)
    bs = 1
    sample = torch.randn(bs, 3, 412, 412)
    pred = model(sample, epoch=20)
    for p in pred:
        print(p.size())