import torch.nn as nn
import torch


class RnnLoss(object):
    def __init__(self, target_delay):
        super(RnnLoss, self).__init__()
        self.target_delay = target_delay
        self.ce_loss = nn.CrossEntropyLoss()

    def __call__(self, predictions, targets):
        losses = 0
        batch_size = predictions.shape[0]
        for pred, target in zip(predictions, targets):
            loss = self.ce_loss(pred[self.target_delay:],
                                target[self.target_delay:])
            losses += loss
        loss = losses / batch_size

        return loss
