import torch
from torch.optim.lr_scheduler import LambdaLR
import torchvision.utils as vutils
import torch.distributed as dist

import errno
import os
import re
import sys
import numpy as np
from bisect import bisect_right


def dict2cuda(data: dict):
    new_dic = {}
    for k, v in data.items():
        if isinstance(v, dict):
            v = dict2cuda(v)
        elif isinstance(v, torch.Tensor):
            v = v.cuda()
        new_dic[k] = v
    return new_dic

def dict2numpy(data: dict):
    new_dic = {}
    for k, v in data.items():
        if isinstance(v, dict):
            v = dict2numpy(v)
        elif isinstance(v, torch.Tensor):
            v = v.detach().cpu().numpy().copy()
        new_dic[k] = v
    return new_dic

def dict2float(data: dict):
    new_dic = {}
    for k, v in data.items():
        if isinstance(v, dict):
            v = dict2float(v)
        elif isinstance(v, torch.Tensor):
            v = v.detach().cpu().item()
        new_dic[k] = v
    return new_dic

def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    """ Create a schedule with a learning rate that decreases linearly after
    linearly increasing during a warmup period.
    """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)

def get_step_schedule_with_warmup(optimizer, milestones, gamma=0.1, warmup_factor=1.0/3, warmup_iters=500, last_epoch=-1,):
    def lr_lambda(current_step):
        if current_step < warmup_iters:
            alpha = float(current_step) / warmup_iters
            current_factor = warmup_factor * (1. - alpha) + alpha
        else:
            current_factor = 1.

        return max(0.0,  current_factor * (gamma ** bisect_right(milestones, current_step)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)

def add_summary(data_items: list, logger, step: int, flag: str, max_disp=4):
    for data_item in data_items:
        tags = data_item['tags']
        vals = data_item['vals']
        dtype = data_item['type']
        if dtype == 'points':
            for i in range(min(max_disp, len(tags))):
                logger.add_mesh('{}/{}'.format(flag, tags[i]),
                                vertices=vals[0], colors=vals[1], global_step=step)
        elif dtype == 'scalars':
            for tag, val in zip(tags, vals):
                if val == 'None':
                    val = 0
                logger.add_scalar('{}/{}'.format(flag, tag),
                                  val, global_step=step)
        else:
            raise NotImplementedError

class DictAverageMeter(object):
    def __init__(self):
        self.data = {}

    def update(self, new_input: dict):
        for k, v in new_input.items():
            if isinstance(v, list):
                self.data[k] = self.data.get(k, []) + v
            else:
                assert (isinstance(v, float) or isinstance(v, int)), type(v)
                self.data[k] = self.data.get(k, []) + [v]

    def mean(self):
        ret = {}
        for k, v in self.data.items():
            if not v:
                ret[k] = 'None'
            else:
                ret[k] = np.round(np.mean(v), 4)
        return ret

    def reset(self):
        self.data = {}

def calc_stat(sample, prob, scores, label_type='label', ignore_id=255):
    T2L = lambda x: x.float().detach().cpu().numpy().tolist()

    labels = sample[label_type]
    max_probs, preds = torch.max(prob, dim=1, keepdim=False)

    # remove ignore cases
    valid_inds = labels != ignore_id
    labels = labels[valid_inds]
    max_probs = max_probs[valid_inds]
    preds = preds[valid_inds]
    #

    all_acc = torch.mean((preds == labels).float()).item()
    scores.update({'{}_all_acc'.format(label_type): all_acc})

    for i in range(2):
        pst_inds = (preds == i)
        if torch.sum(pst_inds) > 0:
            precision = T2L(preds[pst_inds] == labels[pst_inds])
        else:
            precision = []
        scores.update({'{}_precision_{}'.format(label_type, i): precision})

    for thresh in [0.1, 0.4]:
        sel_inds = (torch.abs(max_probs-0.5) > thresh)
        ratio = torch.mean(sel_inds.float()).item()
        if ratio > 0:
            th_acc = T2L(preds[sel_inds] == labels[sel_inds])
        else:
            th_acc = []

        scores.update({'{}_P{}_ratio'.format(label_type, thresh): ratio,
                       '{}_P{}_acc'.format(label_type, thresh): th_acc,
                       })