
import torch
import torch.nn as nn

from dllib.models.building_blocks.dropblock_layers import *
from dllib.models.building_blocks.util_layers import *
from dllib.models.builder.parse_model import parse_model_from_cfg
from dllib.utils.model_utils import model_info


class BuildClassificationHead(nn.Module):
    def __init__(self, cfg, info=True):
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
        self.nc = self.yaml["nc"]

        self.model = parse_model_from_cfg(self.yaml, self.mode)
        self.output_layers = self.yaml["output_layers"]
        if not self.output_layers:
            self.output_layers = [x for x in self.yaml["output_layers"] if x < 0]

        if info:
            self.info(verbose=False)
            pass

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
        result = [output[t] for t in sorted(self.output_layers)]
        return result


if __name__ == "__main__":
    cfg = "../cfgs/base_classification_head.yaml"
    head = BuildClassificationHead(cfg, info=True)

    bs = 12
    sample = [
        torch.randn(bs, 72, 6, 6)
    ]

    pred = head(sample)
    print(pred)
    print(pred[0].shape)
