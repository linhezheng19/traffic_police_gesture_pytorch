import torch
import os
import numpy as np
import argparse

import _init_paths
from libs.models.rnn import rnn_net
from libs.detectors.rnn_detector import RnnDetector, Detect
from libs.config import cfg, merge_cfg_from_file
from libs.utils.utils import load_weights


parser = argparse.ArgumentParser()
parser.add_argument("--cfg", help="optional cfg file", default="cfgs/rnn.yaml")
parser.add_argument("--files", help="run rnn detect on .npy files which generate from pose net",
                    default=False, action="store_true")
parser.add_argument("--dir", help="optional path of .npy files", default=None)
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
    model = rnn_net.RnnNet(cfg.RNN.DIM_IN)
    load_weights(model, cfg.TEST.WEIGHTS)
    model.eval()
    model = model.to(device)

    detector = RnnDetector()
    detect = Detect(model, detector, device)
    if args.files:
        data_dir = args.dir if args.dir is not None else \
            cfg.DATA_DIR + "/" + cfg.TEST.NPY_DIR
        ckpt_dir = cfg.CKPT + "/" + cfg.TEST.SAVE_PATH
        os.makedirs(ckpt_dir, exist_ok=True)
        detect.save_labels(data_dir, ckpt_dir)


if __name__ == "__main__":
    main()