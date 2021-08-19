
import torch
import torch.nn as nn

from dllib.models.building_blocks.dropblock_layers import *
from dllib.models.building_blocks.util_layers import *
from dllib.models.builder.parse_model import parse_model_from_cfg
from dllib.utils.model_utils import model_info


class BuildDetectionHead(nn.Module):
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
        self.strides = self.yaml["strides"]
        self.nc = self.yaml["nc"]
        self.no = self.nc + 5
        anchors = self.yaml["anchors"]
        self.nl = len(anchors)
        self.na = len(anchors[0]) // 2
        self.grid = [torch.zeros(1)] * self.nl
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer("anchors", a)
        self.register_buffer("anchor_grid", a.clone().view(self.nl, 1, -1, 1, 1, 2))

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

        for i in range(self.nl):
            bs, _, ny, nx = result[i].shape
            result[i] = result[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
            result[i][..., :5] = result[i][..., :5].sigmoid()
            #result[i][..., 5:] = result[i][..., 5:].softmax(-1)

            if self.grid[i].shape[2:4] != result[i].shape[2:4]:
                self.grid[i] = self._make_grid(nx, ny)
            self.grid[i] = self.grid[i].to(result[i].device)
            result[i][..., 0:2] = (result[i][..., 0:2] + self.grid[i]) * self.strides[i]
            result[i][..., 2:4] = (result[i][..., 2:4]) * self.anchor_grid[i]

        return result

    @staticmethod
    def _make_grid(nx, ny):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()


if __name__ == "__main__":
    cfg = "../cfgs/base_detection_head_m.yaml"
    head = BuildDetectionHead(cfg, info=True)

    bs = 1
    sample = [
        torch.randn(bs, 256, 52, 52),
        torch.randn(bs, 512, 26, 26),
        torch.randn(bs, 1024, 13, 13),
    ]

    pred = head(sample)
    for i in range(len(pred)):
        bs, _, _, _, info = pred[i].shape
        print(pred[i].shape)
        pred[i] = pred[i].view(bs, -1, info)
    pred = torch.cat(pred, 1)
    print("")
    print(pred.shape)

    from dllib.utils.bbox_utils import non_maximum_suppression
    pred = non_maximum_suppression(pred)
    print("")
    for p in pred:
        print(p)