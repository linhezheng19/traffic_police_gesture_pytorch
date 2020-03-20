import torch
import os
import shutil


def load_weights(model, weights_path):
    try:
        weights_dict = torch.load(weights_path, map_location=torch.device("cpu"))['model']
    except:
        weights_dict = torch.load(weights_path, map_location=torch.device("cpu"))

    model.load_state_dict(weights_dict)
    print("Load weights from {}.".format(weights_path))


def load_optimizer(optimizer, weights_path):
    optimizer_dict = torch.load(weights_path, map_location=torch.device("cpu"))['optimizer']
    optimizer.load_state_dict(optimizer_dict)
    print("Load optimizer from {}.".format(weights_path))


def load_lr_scheduler(lr_scheduler, weights_path):
    lr_scheduler.cur_iter = torch.load(weights_path, map_location=torch.device("cpu"))['lr_scheduler']['cur_iter']
    print("Load lr_scheduler from {}.".format(weights_path))


def save_model(model, ckpt, optimizer=None, lr_scheduler=None):
    state_dict = {'model': model.state_dict()}
    if optimizer is not None:
        state_dict['optimizer'] = optimizer.state_dict()
    if lr_scheduler is not None:
        state_dict['lr_scheduler'] = lr_scheduler.state_dict()
    save_path = os.path.join(ckpt, "latest.pth")
    torch.save(state_dict, save_path)
    print("Latest model have been saved to: {}.".format(save_path))


def save_best_model(model, ckpt, best_iter, cur_iter, best_acc, acc,
                    optimizer=None, lr_scheduler=None, remove_old=True):
    state_dict = {'model': model.state_dict()}
    if optimizer is not None:
        state_dict['optimizer'] = optimizer.state_dict()
    if lr_scheduler is not None:
        state_dict['lr_scheduler'] = lr_scheduler.state_dict()
    best_path = os.path.join(ckpt, "{}iter_{:.2f}.pth".format(cur_iter, acc))
    old_path = os.path.join(ckpt, "{}iter_{:.2f}.pth".format(best_iter, best_acc))
    torch.save(state_dict, best_path)
    if remove_old and best_iter != 0:
        # skip the first save
        os.remove(old_path)
    print("The best model have been saved to: {}.".format(best_path))


def to_one_hot(labels, num_sample, num_classes):
    """Concert label to one_hot type."""
    zeros = torch.zeros(num_sample, num_classes)
    one_hot = zeros.scatter_(1, labels.long(), 1)

    return one_hot


def join_path(first_path, last_path):
    path = os.path.join(first_path, last_path)
    return path


def compute_accuracy(predictions, labels):
    # predictions = predictions.permute(1, 0, 2)
    pred_cls = torch.argmax(predictions, dim=-1)
    # labels = torch.argmax(labels, dim=-1)
    correct = (pred_cls == labels).float()
    # print(pred_cls, labels)
    accuracy = torch.mean(correct)

    return accuracy.data


def compute_time(start_time, end_time):
    all_seconds = end_time - start_time
    seconds = all_seconds % 60
    minutes = (all_seconds // 60) % 60
    hours = all_seconds // 3600
    time_str = "{}:{}:{}".format(int(hours), int(minutes), int(seconds))

    return time_str
