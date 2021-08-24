
import torch
import torch.nn as nn

from dllib.models.building_blocks.dropblock_layers import *
from dllib.models.building_blocks.util_layers import *
from dllib.models.builder.parse_model import parse_model_from_cfg
from dllib.utils.model_utils import model_info


class BBoxNet(nn.Module):
    def __init__(self, cfg, info=False):
        super().__init__()
        self.mode = "bbox head"
        if isinstance(cfg, dict):
            self.yaml = cfg
        else:
            import yaml
            from pathlib import Path
            self.yaml_file = Path(cfg).name
            with open(cfg) as f:
                self.yaml = yaml.safe_load(f)

        self.input_shape = self.yaml["expected_input_shape"]
        self.stride = self.yaml["strides"][1]
        self.w, self.h = self.input_shape[1][1] * self.stride, self.input_shape[1][2] * self.stride
        self.grid = torch.zeros(1)

        self.model = parse_model_from_cfg(self.yaml, self.mode)
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
            elif isinstance(layer, Upsample) or isinstance(layer, Downsample):
                tmp = output[layer.target_layer]
                x = layer(tmp)
            elif isinstance(layer, GetLayer):
                x = output[layer.target_layer]
            else:
                x = layer(x)
            output.append(x)

        bs, no, ny, nx = output[-1].shape
        result = output[-1].view(bs, 1, no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
        if self.grid.shape[1:3] != result.shape[1:3]:
            self.grid = self._make_grid(nx, ny)
        self.grid = self.grid.to(result.device)
        result[..., 0:2] = (result[..., 0:2] + self.grid) * self.stride
        result[..., 2:3] = (result[..., 2:3]) * self.w
        result[..., 3:4] = (result[..., 3:4]) * self.h
        return result

    @staticmethod
    def _make_grid(nx, ny):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, ny, nx, 2)).float()


class ClsNet(nn.Module):
    def __init__(self, cfg, info=False):
        super().__init__()
        self.mode = "cls head"
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

        self.model = parse_model_from_cfg(self.yaml, self.mode)
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
            elif isinstance(layer, Upsample) or isinstance(layer, Downsample):
                tmp = output[layer.target_layer]
                x = layer(tmp)
            elif isinstance(layer, GetLayer):
                x = output[layer.target_layer]
            else:
                x = layer(x)
            output.append(x)
        bs, no, ny, nx = output[-1].shape
        result = output[-1].view(bs, 1, no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
        return result

class BuildSeparateDetectionHead(nn.Module):
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

        bbox_yaml, cls_yaml = self.yaml.copy(), self.yaml.copy()
        bbox_yaml["architecture"] = bbox_yaml["bbox_architecture"]
        cls_yaml["architecture"] = cls_yaml["cls_architecture"]
        self.bboxnet = BBoxNet(bbox_yaml, False)
        self.clsnet = ClsNet(cls_yaml, False)
        self.nc = self.clsnet.nc

    def forward(self, xs, epoch=None):
        output = xs
        bbox = self.bboxnet(output.copy(), epoch)
        cls = self.clsnet(output.copy(), epoch)
        result = torch.cat([bbox, cls], dim=-1)
        return [result]


if __name__ == "__main__":
    cfg = "../cfgs/base_detection_head2.yaml"
    head = BuildSeparateDetectionHead(cfg, info=True)

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
    print(pred.shape)
    print(pred[..., :5])
    print(torch.sum(pred[..., 5:]))

    from dllib.utils.bbox_utils import non_maximum_suppression
    pred = non_maximum_suppression(pred)
    print("")
    for p in pred:
        print(p)