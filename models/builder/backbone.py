
import torch
import torch.nn as nn

from dllib.models.building_blocks.dropblock_layers import *
from dllib.models.building_blocks.util_layers import *
from dllib.models.builder.parse_model import parse_model_from_cfg
from dllib.utils.model_utils import model_info


class BuildBackbone(nn.Module):
    def __init__(self, cfg, info=False):
        super().__init__()
        self.mode = "backbone"
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
        self.output_layers = [x - 1 for x in self.yaml["output_layers"] if x > 0]
        if not self.output_layers:
            self.output_layers = [x for x in self.yaml["output_layers"] if x < 0]

        if info:
            self.info(verbose=False)

    def info(self, verbose=False, batch_size=1):
        model_info(self, verbose, self.input_shape, batch_size)

    def forward(self, x, epoch=None):
        output = []
        for i, layer in enumerate(self.model):
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
        result = [output[t] for t in sorted(self.output_layers)]
        return result


if __name__ == "__main__":
    cfg = "../cfgs/base_backbone_m.yaml"
    bb = BuildBackbone(cfg, info=True)

    bs = 1
    sample = torch.randn(bs, 3, 412, 412)

    pred = bb(sample, epoch=20)
    for p in pred:
        print(p.size())
