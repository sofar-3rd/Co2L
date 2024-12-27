'''
The original source code can be found in
https://github.com/HobbitLong/SupContrast/blob/master/util.py
'''

from __future__ import print_function

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Sampler
from scipy.stats import multivariate_normal
from datetime import datetime, timedelta
from sklearn.metrics import confusion_matrix


class TwoCropTransform:
    """Create two crops of the same image"""

    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def f1_score(precision, recall):
    return 2 * precision * recall / float(precision + recall)


def get_model_stats(y, y_pred):
    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
    acc = (tp + tn) / float(tp + tn + fp + fn)
    fpr = fp / float(fp + tn)
    tpr = tp / float(tp + fn)
    tnr = tn / float(fp + tn)
    fnr = fn / float(fn + tp)
    precision = tp / float(tp + fp)
    recall = tp / float(tp + fn)
    return tpr, tnr, fpr, fnr, acc, precision, f1_score(precision, tpr)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def adjust_learning_rate(args, optimizer, epoch):
    lr = args.learning_rate
    if args.cosine:
        eta_min = lr * (args.lr_decay_rate ** 3)
        lr = eta_min + (lr - eta_min) * (
                1 + math.cos(math.pi * epoch / args.epochs)) / 2
    else:
        steps = np.sum(epoch > np.asarray(args.lr_decay_epochs))
        if steps > 0:
            lr = lr * (args.lr_decay_rate ** steps)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def warmup_learning_rate(args, epoch, batch_id, total_batches, optimizer):
    if args.warm and epoch <= args.warm_epochs:
        p = (batch_id + (epoch - 1) * total_batches) / \
            (args.warm_epochs * total_batches)
        lr = args.warmup_from + p * (args.warmup_to - args.warmup_from)

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


def set_optimizer(opt, model):
    optimizer = optim.SGD(model.parameters(),
                          lr=opt.learning_rate,
                          momentum=opt.momentum,
                          weight_decay=opt.weight_decay)
    return optimizer


def set_cls_optimizer(opt, model):
    optimizer = optim.SGD(model.parameters(),
                          lr=opt.cls_learning_rate,
                          momentum=opt.cls_momentum,
                          weight_decay=opt.cls_weight_decay)
    return optimizer


def save_model(model, optimizer, opt, epoch, save_file):
    print('==> Saving...' + save_file)
    state = {
        'opt': opt,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
    }
    torch.save(state, save_file)
    del state


def load_model(model, optimizer, save_file):
    print('==> Loading...' + save_file)
    loaded = torch.load(save_file)

    model.load_state_dict(loaded['model'])
    optimizer.load_state_dict(loaded['optimizer'])
    del loaded

    return model, optimizer


def generate_months(start_month, end_month):
    # 将输入的月份字符串转换为日期对象
    start_date = datetime.strptime(start_month, "%Y-%m")
    end_date = datetime.strptime(end_month, "%Y-%m")

    # 创建一个空的月份列表
    months_list = []

    # 从开始月份到结束月份，逐月添加到列表
    while start_date <= end_date:
        months_list.append(start_date.strftime("%Y-%m"))
        # 将日期加上一个月
        start_date = start_date.replace(day=28) + timedelta(days=4)  # 使日期跳转到下一个月的1号

    if months_list[-1] != end_month:
        months_list.append(end_month)

    return months_list


# 返回当前时间字符串
def current_time():
    return datetime.now().strftime("%m%d%H%M")
