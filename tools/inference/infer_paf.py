import torch
import numpy as np
import argparse
import os

import _init_paths
from libs.detectors.paf_detector import PafDetector, Detect
from libs.models.pose.paf import PafNet
from libs.config import cfg, merge_cfg_from_file
from libs.utils.utils import load_weights


parser = argparse.ArgumentParser(description="Pose Detection.")
parser.add_argument("--cfg", help="optional config file", default="cfgs/paf.yaml")
parser.add_argument("--cam", help="using camera", default=False, action="store_true")
parser.add_argument("--h", help="detect heatmaps, must set cam=True first",
                    default=False, action="store_true")
parser.add_argument("--b", help="detect bones, must set cam=True first",
                    default=False, action="store_true")
parser.add_argument("--files", help="detect video files and save features",
                    default=False, action="store_true")
parser.add_argument("--dir", help="video files path for a single file", default=None)
parser.add_argument("--gpu_id", help="which gpu used", type=int, default=0)

args = parser.parse_args()
if args.cfg is not None:
    merge_cfg_from_file(args.cfg)

if cfg.DEVICE == 'cuda':
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)


def main():
    # set device
    device = torch.device(cfg.DEVICE)
    # creat eval model
    if cfg.POSE.TYPE == 'paf':
        model = PoseNet(cfg.POSE.NUM_PAFS, cfg.POSE.NUM_KPTS)
    load_weights(model, cfg.TEST.WEIGHTS)
    model.eval()
    model.to(device)
    detector = PoseDetector(cfg.POSE.NUM_PAFS, cfg.POSE.NUM_KPTS)
    detect = Detect(model, detector, device, cfg.POSE.SCALE, cfg.POSE.KPT_THRESH)
    if args.cam:
        if args.b:
            detect.cam_det_bones()
        elif args.h:
            detect.cam_det_heatmaps()
    elif args.files:
        data_dir = args.dir if args.dir is not None else \
            cfg.DATA_DIR + "/" + cfg.TEST.DATA_PATH
        # save the paf feature to data path for rnn training/testing
        save_dir = cfg.DATA_DIR + "/" + cfg.TEST.SAVE_PATH
        os.makedirs(save_dir, exist_ok=True)
        detect.save_features(data_dir, save_dir)


if __name__ == "__main__":
    main()
