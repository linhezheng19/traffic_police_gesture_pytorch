import torch
import torch.nn as nn
import time
import os

from libs.utils.utils import compute_accuracy, save_model, \
    save_best_model, compute_time


class PafTrainer(object):
    def __init__(self):
        super(PafTrainer, self).__init__()

    def __call__(self, model, loader, loss_func, optimizer,
                 eval_iter, ckpt, device, lr_scheduler=None):
        start_time = time.time()
        log_file = os.path.join(ckpt, "log.txt")
        # log = open(log_file, "w+")
        train_loader, test_loader = loader
        val_loss = 0
        with open(log_file, 'w') as f:
            for i, (_, imgs, targets) in enumerate(train_loader):
                if lr_scheduler is not None:
                    cur_iter = lr_scheduler.cur_iter + 1
                    lr_scheduler.step(cur_iter)  # adjust learning rate
                else:
                    cur_iter = i + 1
                optimizer.zero_grad()  # zeros the gradient

                outputs = model(imgs.to(device))
                loss = loss_func(outputs, targets)

                loss.backward()
                optimizer.step()
                print("LOSS: {:.8f}     ITER: {}      VAL_LOSS: {}"
                      .format(loss.data, cur_iter, val_loss))
                # log the loss and iteration

                if cur_iter % eval_iter == 0:
                    # save the checkpoint
                    print("Computing loss on test data set ...")
                    # val_loss = self._test(model, test_loader, loss_func, device)
                    save_model(model, ckpt, optimizer, lr_scheduler)
                f.write("LOSS: {:.8f}     ITER: {}      VAL_LOSS: {}\n"
                      .format(loss.data, cur_iter, val_loss))

        end_time = time.time()
        cost_time = compute_time(start_time, end_time)
        print("Training is done, cost: {}.".format(cost_time))

        return None

    def _test(self, model, loader, loss_func, device):
        # It will enumerate all data in test dataset, which cost some time, so not used necessarily.
        model.eval()
        val_loss = 0
        for i, (_, feature, targets) in enumerate(loader):
            i += 1
            outputs = model(feature.to(device))
            loss = loss_func(outputs, targets)
            val_loss += loss

        model.train()  # switch to train mode, or will raise error
        return val_loss / i
