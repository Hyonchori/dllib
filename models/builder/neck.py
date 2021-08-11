
import torch
import torch.nn as nn

from dllib.models.builder.parse_model import parse_model_from_cfg
from dllib.utils.model_utils import model_info


class BuildNeck(nn.Module):
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

        print(self.yaml)
        self.input_shape = self.yaml["expected_input_shape"]


if __name__ == "__main__":
    cfg = "../cfgs/base_neck_m.yaml"
    neck = BuildNeck(cfg, info=True)

    bs = 1
    sample = [
        torch.randn(bs, 128, 52, 52),
        torch.randn(bs, 256, 26, 26),
        torch.randn(bs, 512, 13, 13),
    ]