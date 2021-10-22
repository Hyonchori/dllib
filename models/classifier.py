
import torch
import torch.nn as nn

from dllib.models.builder.backbone import BuildBackbone
from dllib.models.builder.neck import BuildNeck
from dllib.models.builder.classification_head import BuildClassificationHead
from dllib.utils.model_utils import model_info


class BuildClassifier(nn.Module):
    def __init__(self,
                 backbone_cfg,
                 neck_cfg,
                 head_cfg,
                 mode: str=None,
                 info: bool=False):
        super().__init__()
        self.backbone = BuildBackbone(backbone_cfg, True)
        self.neck = BuildNeck(neck_cfg, True) if neck_cfg is not None else None
        self.head= BuildClassificationHead(head_cfg, True)

        self.mode = "classifier" if mode is None else mode
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

    bb_cfg = "cfgs/backbone_clf.yaml"
    h_cfg = "cfgs/base_classification_head.yaml"
    model = BuildClassifier(bb_cfg,
                            neck_cfg=None,
                            head_cfg=h_cfg,
                            info=True)
    bs = 1
    sample = torch.randn(bs, 3, 128, 128)
    pred = model(sample, epoch=20)
    for p in pred:
        print(p.size())