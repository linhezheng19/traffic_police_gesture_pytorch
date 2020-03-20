import os
import cv2
import argparse
import numpy as np

import _init_paths
from libs.utils.datasets import load_labels
from libs.config import cfg, merge_cfg_from_file
from libs.utils.video_utils import save_single_video, save_all_videos


parser = argparse.ArgumentParser()
parser.add_argument("--cfg", help="optional cfg file", default="cfgs/rnn.yaml")
parser.add_argument("--v", help="video file path, directory or file", type=str, default=None)
parser.add_argument("--c", help="csv file path, generally a directory", type=str, default=None)
parser.add_argument("--o", help="video saving path", type=str)
parser.add_argument("--t", help="video type, mp4/avi/...", type=str, default="mp4")

args = parser.parse_args()

if args.cfg is not None:
    merge_cfg_from_file(args.cfg)


def main():
    gestures = cfg.RNN.GESTURES

    os.makedirs(args.o, exist_ok=True)

    if os.path.isfile(args.v):
        save_single_video(args.v, args.c, args.o, gestures)
        print("Video has been saved at {}.".format(args.v))
    else:
        save_all_videos(args.v, args.c, args.o, gestures, args.t)
        print("All videos have been saved at {}.".format(args.v))


if __name__ == "__main__":
    main()
