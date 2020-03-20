import torch
import torch.nn as nn
import time

from libs.utils.utils import compute_accuracy, save_model, \
    save_best_model, compute_time


class RnnTrainer(object):
    def __init__(self):
        super(RnnTrainer, self).__init__()

    def __call__(self, model, loader, loss_func, optimizer,
                 eval_iter, ckpt, device, lr_scheduler=None):
        start_time = time.time()
        train_loader, test_loader = loader
        best_acc = 0
        best_iter = 0
        for i, (feature, labels) in enumerate(train_loader):
            if lr_scheduler is not None:
                cur_iter = lr_scheduler.cur_iter + 1
                lr_scheduler.step(cur_iter)  # adjust learning rate
            else:
                cur_iter = i + 1
            optimizer.zero_grad()  # zeros the gradient

            outputs = model(feature.to(device))
            loss = loss_func(outputs, labels.to(device))

            loss.backward()
            optimizer.step()
            print("LOSS: {:.8f}     ITER: {}".format(loss.data, cur_iter))
            if cur_iter % eval_iter == 0:
                print("Evaluating model in test dataset ...")
                acc = self._test(model, test_loader, device)
                # save the checkpoint and the best acc model
                save_model(model, ckpt, optimizer, lr_scheduler)

                if acc >= best_acc:
                    save_best_model(model, ckpt, best_iter, cur_iter, best_acc,
                                    acc, optimizer, lr_scheduler)
                    best_acc = acc
                    best_iter = cur_iter
                print("The best result is: {:.2f}%.".format(best_acc))
        end_time = time.time()
        cost_time = compute_time(start_time, end_time)
        print("Training is done, cost: {}.".format(cost_time))

        return None

    def _test(self, model, loader, device):
        model.eval()
        accuracy = 0
        num_iter = 0
        for i, (feature, labels) in enumerate(loader):
            outputs = model(feature.to(device))
            acc = compute_accuracy(outputs, labels.to(device))
            accuracy += acc
            num_iter += 1
        accuracy = accuracy / num_iter * 100  # convert to %
        print("The test accuracy is: {:.2f}%.".format(accuracy))

        model.train()  # switch to train mode, or will raise error

        return accuracy
