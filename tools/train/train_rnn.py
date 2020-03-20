import torch
import os
import argparse

from torch.utils.data import DataLoader

import _init_paths
from libs.models.rnn.rnn_net import RnnNet
from libs.trainers.rnn_trainer import RnnTrainer
from libs.utils.sampler import make_batch_sampler
from libs.utils.solver import LrScheduler, Optimizer
from libs.utils.datasets import NpyDataset
from libs.models.loss.rnn_loss import RnnLoss
from libs.config import cfg, merge_cfg_from_file
from libs.utils.utils import join_path, load_weights, \
    load_optimizer, load_lr_scheduler


parser = argparse.ArgumentParser()
parser.add_argument("--cfg", help="optional cfg file", default='cfgs/rnn.yaml')
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
    train_npy_dir = join_path(cfg.DATA_DIR, cfg.TRAIN.NPY_DIR)
    train_csv_dir = join_path(cfg.DATA_DIR, cfg.TRAIN.CSV_DIR)
    test_npy_dir = join_path(cfg.DATA_DIR, cfg.TEST.NPY_DIR)
    test_csv_dir = join_path(cfg.DATA_DIR, cfg.TEST.CSV_DIR)

    # creat model
    model = RnnNet(cfg.RNN.DIM_IN)
    model.to(device)

    optimizer = Optimizer(model, cfg).build_optimizer()
    lr_schedule = LrScheduler(cfg, optimizer)

    if args.weights is not None:
        # only support resume now and you should input weight path
        load_weights(model, args.weights)
        load_optimizer(optimizer, args.weights)
        load_lr_scheduler(lr_schedule, args.weights)

    # make train/test dataset and dataloader
    train_dataset = NpyDataset(train_npy_dir, train_csv_dir, cfg.RNN.TIME_STEP,
                               cfg.RNN.TARGET_DELAY, cfg.RNN.NUM_CLASSES)
    test_dataset = NpyDataset(test_npy_dir, test_csv_dir, cfg.RNN.TIME_STEP,
                              cfg.RNN.TARGET_DELAY, cfg.RNN.NUM_CLASSES)
    train_batch_sampler = make_batch_sampler(train_dataset, cfg.TRAIN.BATCH_SIZE,
                                             cfg.TRAIN.ITERS)
    train_loader = DataLoader(train_dataset,
                              num_workers=cfg.TRAIN.LOAD_THREADS,
                              batch_sampler=train_batch_sampler,
                              )
    test_loader = DataLoader(test_dataset,
                             batch_size=cfg.TRAIN.BATCH_SIZE,
                             shuffle=True,
                             num_workers=cfg.TRAIN.LOAD_THREADS,
                             pin_memory=True,
                             )
    # create loss function
    loss_func = RnnLoss(cfg.RNN.TARGET_DELAY)
    # create trainer
    train = RnnTrainer()
    # creat checkpoint dir
    os.makedirs(cfg.CKPT, exist_ok=True)
    # start training
    train(model, (train_loader, test_loader),loss_func, optimizer,
          cfg.TRAIN.SNAPSHOT, cfg.CKPT, device, lr_schedule)


if __name__ == "__main__":
    main()
