
import torch
import torch.nn as nn

from dllib.models.builder.backbone import BuildBackbone
from dllib.models.builder.neck import BuildNeck
from dllib.models.builder.detection_head import BuildDetectionHead
from dllib.models.builder.detection_head_separate import BuildSeparateDetectionHead
from dllib.models.builder.keypoint_head import BuildKeypointDetectionHead
from dllib.utils.model_utils import model_info


class BuildDetector(nn.Module):
    def __init__(self,
                 backbone_cfg,
                 neck_cfg,
                 detector_head_cfg,
                 head_mode: str="separate",
                 mode: str=None,
                 info: bool=False):
        super().__init__()
        self.backbone = BuildBackbone(backbone_cfg, True)
        self.neck = BuildNeck(neck_cfg, True) if neck_cfg is not None else None
        if head_mode == "separate":
            self.head = BuildSeparateDetectionHead(detector_head_cfg, True)
        elif head_mode == "unified":
            self.head = BuildDetectionHead(detector_head_cfg, True)
        elif head_mode == "keypoint":
            self.head = BuildKeypointDetectionHead(detector_head_cfg, True)

        self.mode = "detector" if mode is None else mode
        self.input_shape = self.backbone.input_shape
        if info:
            self.stride = self.backbone.stride
            self.info(verbose=False)

    def info(self, verbose=False, batch_size=1):
        return model_info(self, verbose, self.input_shape, batch_size)

    def forward(self, x, epoch=None):
        x = self.backbone(x, epoch)
        x = self.neck(x, epoch) if self.neck is not None else x
        x = self.head(x, epoch)
        return x


if __name__ == "__main__":

    # separate object detector
    '''bb_cfg = "cfgs/base_backbone_m.yaml"
    n_cfg = "cfgs/base_neck_m.yaml"
    h_cfg = "cfgs/base_detection_head2.yaml"
    model = BuildDetector(bb_cfg,
                          n_cfg,
                          h_cfg,
                          head_mode="separate",
                          True)
    bs = 1
    sample = torch.randn(bs, 3, 416, 416)
    pred = model(sample, epoch=20)
    for p in pred:
        print(p.size())'''

    # keypoint detector
    bb_cfg = "cfgs/base_backbone_m.yaml"
    n_cfg = "cfgs/base_neck_m.yaml"
    h_cfg = "cfgs/base_detection_head2.yaml"
    model = BuildDetector(bb_cfg,
                          n_cfg,
                          h_cfg,
                          head_mode="separate",
                          info=True)
    bs = 1
    sample = torch.randn(bs, 3, 416, 416)
    pred = model(sample, epoch=20)
    for p in pred:
        print(p.size())