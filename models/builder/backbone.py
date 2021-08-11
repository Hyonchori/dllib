
import torch
import torch.nn as nn

from dllib.models.building_blocks.conv_layers import *
from dllib.models.building_blocks.attention_layers import *
from dllib.models.building_blocks.dropblock_layers import *
from dllib.models.building_blocks.util_layers import *
from dllib.utils.general_utils import get_module_name, make_divisible
from dllib.utils.model_utils import model_info


class BuildBackbone(nn.Module):
    def __init__(self, cfg, info=False):
        super().__init__()
        if isinstance(cfg, dict):
            self.yaml = cfg
        else:
            import yaml
            from pathlib import Path
            self.yaml_file = Path(cfg).name
            with open(cfg) as f:
                self.yaml = yaml.safe_load(f)

        self.input_shape = self.yaml["expected_input_shape"]
        self.model = parse_backbone_from_cfg(self.yaml)
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
            else:
                x = layer(x)
            output.append(x)
        result = [output[t] for t in sorted(self.output_layers)]
        return result


def parse_backbone_from_cfg(cfg):  # cfg: dict
    print('\n%5s%10s   %-60s%-30s' % ('idx', 'params', 'module', 'arguments'))
    cm = cfg["channel_multiple"]
    layers = []
    for i, (m, args) in enumerate(cfg["architecture"]):
        m = eval(m) if isinstance(m, str) else m
        if m in [ConvBnAct, DWConv, PWConv, DWSConv, ResidualBlock, Bottleneck, FusedBottleneck,
                 FusedBottleneck, BottleneckCSP, SPP, Focus, ConvDownSampling]:
            if i != 0:
                args[0] = make_divisible(args[0] * cm, 8)
                args[1] = make_divisible(args[1] * cm, 8)
            else:
                args[1] = make_divisible(args[1] * cm, 8)

        module = m(*args)
        module_name = get_module_name(m)
        num_params = sum([x.numel() for x in module.parameters()])

        print("%5s%10.0f   %-60s%-30s" % (i, num_params, module_name, args))
        layers.append(module)
    print("")
    return nn.Sequential(*layers)


if __name__ == "__main__":
    import time

    cfg = "../cfgs/base_backbone_m.yaml"
    bb = BuildBackbone(cfg, info=True)

    bs = 1
    sample = torch.randn(bs, 3, 412, 412)
    t1 = time.time()
    pred = bb(sample, epoch=20)
    print(time.time() - t1)
    for p in pred:
        print(p.size())
