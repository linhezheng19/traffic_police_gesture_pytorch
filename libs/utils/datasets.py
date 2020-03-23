import numpy as np
import torch
import glob
import os
import cv2
import json

from torch.utils.data import Dataset

from libs.utils.utils import to_one_hot
from libs.utils.data_utils import extract_length_angle, load_labels, \
    clip_npy_labels_by_time_step, load_features_labels, delay
from libs.utils.augments import pad_resize, resize_annos, ia_augment


class NpyDataset(Dataset):
    # Dataset from csv files to train/test rnn network.
    def __init__(self, npy_dir, csv_dir, time_step, target_delay, num_classes):
        super(NpyDataset, self).__init__()
        self.features, self.labels = load_features_labels(
                                    npy_dir, csv_dir, time_step)
        self.target_delay = target_delay
        self.num_classes = num_classes
        self.time_step = time_step

    def __getitem__(self, index):
        feature = self.features[index]
        feature = extract_length_angle(feature.numpy())
        feature = torch.from_numpy(feature)
        label = delay(self.labels[index], self.target_delay)
        # label = label.view(-1, 1)
        # label = to_one_hot(label, self.time_step, self.num_classes)

        return feature, label.long()

    def __len__(self):
        return len(self.features)


class AIChallenger(Dataset):
    # AIChallenger keypoints dataset.
    def __init__(self, anno_file, data_path, im_size, augments=True):
        super(AIChallenger, self).__init__()
        f = open(anno_file)
        self.annos = json.load(f)
        self.data_path = data_path
        self.im_size = im_size
        self.augments = augments

    def __getitem__(self, index):
        anno = self.annos[index]
        img_name = anno["image_id"]
        kpts_annos = anno["keypoint_annotations"]
        kpts = []
        for human in kpts_annos.keys():
            # reshape to [1, num_kpts, 3], and 3 for x,y, visible
            kpt = np.array(kpts_annos[human]).reshape(-1, 3)[np.newaxis, ...]
            kpts.append(kpt)
        kpts = np.concatenate(kpts, 0)
        img_path = os.path.join(self.data_path, img_name+".jpg")
        img = cv2.imread(img_path)
        img, resize_factors = pad_resize(img, self.im_size)
        kpts = resize_annos(kpts, resize_factors)
        img = img.transpose(2, 0, 1)
        if self.augments:
            img, kpts = ia_augment(img, kpts)

        img = torch.from_numpy(img).float()
        # Handle images with less than three channels
        if len(img.shape) != 3:
            img = img.unsqueeze(0)
            img = img.expand((3, img.shape[1:]))

        return img_path, img, kpts

    def __len__(self):
        return len(self.annos)

    # for vis the img after process
    # def circle(self, img, kpts):
    #     for kpt in kpts:
    #         for joint in kpt:
    #             if joint[2] == 1:
    #                 cv2.circle(img, (joint[0], joint[1]), 10, (255, 255, 255), -1)
    def collate_fn(self, batch):
        """Make the target a batch."""
        paths, imgs, targets = list(zip(*batch))
        targets_list = []
        for target in targets:
            targets_list.append(target)
        imgs = torch.stack(imgs)

        return paths, imgs, targets_list
