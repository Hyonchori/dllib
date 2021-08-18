
import argparse

import torch




def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone_cfg", type=str, help="backbone.yaml path",
                        default="../../models/cfgs/base_backbone_m.yaml")
    parser.add_argument("--neck_cfg", type=str, help="backbone.yaml path",
                        default="../../models/cfgs/base_neck_m.yaml")
    parser.add_argument("--head_cfg", type=str, help="backbone.yaml path",
                        default="../../models/cfgs/base_detection_head_m.yaml")
    parser.add_argument("--weights", type=str, help="initial weights path")
    parser.add_argument("--img_size", type=int, default=412)
    parser.add_argument("--save_dir", type=str, default="../../runs")
    parser.add_argument("--name", type=str, default="base_detector")
    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt




if __name__ == "__main__":
    opt = parse_opt(True)