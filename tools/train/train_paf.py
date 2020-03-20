import torch
import os
import argparse

from torch.utils.data import DataLoader

import _init_paths
from libs.models.pose.paf import PafNet
from libs.trainers.paf_trainer import PafTrainer
from libs.utils.sampler import make_batch_sampler
from libs.utils.solver import LrScheduler, Optimizer
from libs.utils.datasets import AIChallenger
from libs.models.loss.paf_loss import PafLoss
from libs.config import cfg, merge_cfg_from_file
from libs.utils.utils import join_path, load_weights, \
    load_optimizer, load_lr_scheduler


parser = argparse.ArgumentParser()
parser.add_argument("--cfg", help="optional cfg file", default='cfgs/paf.yaml')
parser.add_argument("--gpu_id", help="which gpu used", type=int, default=0)
parser.add_argument("--weights", help="pre-trained model", type=str, default=None)

args = parser.parse_args()

if args.cfg is not None:
    merge_cfg_from_file(args.cfg)

if cfg.DEVICE == 'cuda':
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)


def main():
    # set device
    device = torch.device(cfg.DEVICE)
    # train/test data path
    train_data_dir = join_path(cfg.DATA_DIR, cfg.TRAIN.DATA_PATH)
    train_image_dir = join_path(train_data_dir, "images/train")
    train_anno_dir = join_path(train_data_dir,
                               "annotations/keypoint_train_annotations.json")
    test_image_dir = join_path(train_data_dir, "images/val")
    test_anno_dir = join_path(train_data_dir,
                              "annotations/keypoint_val_annotations.json")

    # creat model
    model = PafNet(cfg.POSE.NUM_PAFS, cfg.POSE.NUM_KPTS)
    model.to(device)

    optimizer = Optimizer(model, cfg).build_optimizer()
    lr_schedule = LrScheduler(cfg, optimizer)

    if args.weights is not None:
        # only support resume now and you should input weight path
        load_weights(model, args.weights)
        load_optimizer(optimizer, args.weights)
        load_lr_scheduler(lr_schedule, args.weights)

    # make train/test dataset and dataloader
    train_dataset = AIChallenger(train_anno_dir, train_image_dir, cfg.TRAIN.SCALE)
    train_batch_sampler = make_batch_sampler(train_dataset, cfg.TRAIN.BATCH_SIZE,
                                             cfg.TRAIN.ITERS, lr_schedule.cur_iter)
    train_loader = DataLoader(train_dataset,
                              num_workers=cfg.TRAIN.LOAD_THREADS,
                              batch_sampler=train_batch_sampler,
                              collate_fn=train_dataset.collate_fn
                              )

    test_dataset = AIChallenger(test_anno_dir, test_image_dir, cfg.TRAIN.SCALE)
    test_loader = DataLoader(test_dataset,
                             num_workers=cfg.TRAIN.LOAD_THREADS,
                             batch_size=cfg.TEST.BATCH_SIZE,
                             shuffle=True,
                             pin_memory=True,
                             collate_fn=test_dataset.collate_fn)

    # create loss function
    loss_func = PafLoss(cfg.POSE.LIMBS, cfg.POSE.LIMB_WIDTH, cfg.TRAIN.SCALE, cfg.POSE.GAUSSIAN_VAR)
    # create trainer
    train = PafTrainer()
    # creat checkpoint dir
    os.makedirs(cfg.CKPT, exist_ok=True)
    # start training
    train(model, (train_loader, test_loader), loss_func, optimizer,
          cfg.TRAIN.SNAPSHOT, cfg.CKPT, device, lr_schedule)


if __name__ == "__main__":
    main()
