
import torch
import torch.nn as nn

from dllib.models.building_blocks.dropblock_layers import *
from dllib.models.building_blocks.util_layers import *
from dllib.models.builder.parse_model import parse_model_from_cfg
from dllib.utils.model_utils import model_info


class BuildKeypointDetectionHead(nn.Module):
    def __init__(self, cfg, info=False):
        super().__init__()
        self.mode = "head"
        if isinstance(cfg, dict):
            self.yaml = cfg
        else:
            import yaml
            from pathlib import Path
            self.yaml_file = Path(cfg).name
            with open(cfg) as f:
                self.yaml = yaml.safe_load(f)

        self.input_shape = self.yaml["expected_input_shape"]

        self.model = parse_model_from_cfg(self.yaml, self.mode)
        if info:
            pass
            self.info(verbose=False)

    def info(self, verbose=False, batch_size=1):
        model_info(self, verbose, self.input_shape, batch_size)

    def forward(self, xs, epoch=None):
        output = xs
        x = None
        for i, layer in enumerate(self.model):
            if x is None:
                x = output[-1]

            if isinstance(layer, LSDropBlock):
                x = layer(x, epoch)
            elif isinstance(layer, ConcatLayer):
                tmp = [output[t] for t in layer.target_layers]
                x = layer(tmp)
            elif isinstance(layer, Upsample):
                tmp = output[layer.target_layer]
                x = layer(tmp)
            elif isinstance(layer, GetLayer):
                x = output[layer.target_layer]
            else:
                x = layer(x)
            output.append(x)
        bs, _, _, _ = output[-1].shape
        result = output[-1].view(bs, -1)
        return result


if __name__ == "__main__":
    cfg = "../cfgs/keypoint_detection_head.yaml"
    head = BuildKeypointDetectionHead(cfg, info=True)

    bs = 1
    sample = [
        torch.randn(bs, 128, 32, 32),
        torch.randn(bs, 256, 16, 16),
        torch.randn(bs, 512, 8, 8),
    ]

    pred = head(sample)
    print(pred.shape)
    print(pred)
    print(head)