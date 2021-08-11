
from dllib.models.building_blocks.conv_layers import *
from dllib.models.building_blocks.attention_layers import *
from dllib.models.building_blocks.dropblock_layers import *
from dllib.models.building_blocks.util_layers import *
from dllib.utils.general_utils import get_module_name, make_divisible


def parse_model_from_cfg(cfg):  # cfg: dict
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