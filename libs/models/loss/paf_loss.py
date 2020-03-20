import torch
import torch.nn as nn
import numpy as np

from libs.utils.targets import keypoints_heatmap, part_affinity_field
from libs.config import cfg


class PafLoss(object):
    def __init__(self, paris, limb_width, im_size, var):
        super(PafLoss, self).__init__()
        self.mse = nn.MSELoss()
        self.pairs = np.array(paris) - 1
        self.limb_width = limb_width
        self.im_size = im_size
        self.var = var

    def __call__(self, inputs, targets=None):
        # inputs: list([b, c, h, w], ...), targets: list([n, 14, 3], ...)
        losses = 0
        for x in inputs:
            for batch_idx in range(x.shape[0]):
                heat_size = x[batch_idx].shape[1:]
                stride = self.im_size[0] // heat_size[1]
                heatmap = keypoints_heatmap(targets[batch_idx], heat_size, stride, self.var)
                paf = part_affinity_field(targets[batch_idx], heat_size, stride, self.pairs, self.limb_width)
                paf_size = paf.shape[0]
                heatmap = torch.from_numpy(heatmap)
                paf = torch.from_numpy(paf)
                if x.is_cuda:
                    x = x.cuda()
                    heatmap = heatmap.cuda()
                    paf = paf.cuda()
                loss1 = self.mse(x[batch_idx][:paf_size], paf)
                loss2 = self.mse(x[batch_idx][paf_size:], heatmap)
                losses += (loss1 + loss2)

        return losses
