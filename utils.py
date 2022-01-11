# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import random
import re
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch import distributions as pyd
from torch.distributions.utils import _standard_normal
import socket
import libtmux
import pygit2 as pg

chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ" #these are the digits you will use for conversion back and forth
charsLen = len(chars)

class eval_mode:
    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(False)

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False


def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data +
                                (1 - tau) * target_param.data)


def to_torch(xs, device):
    return tuple(torch.as_tensor(x, device=device) for x in xs)


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data, gain)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)


class Until:
    def __init__(self, until, action_repeat=1):
        self._until = until
        self._action_repeat = action_repeat

    def __call__(self, step):
        if self._until is None:
            return True
        until = self._until // self._action_repeat
        return step < until


class Every:
    def __init__(self, every, action_repeat=1):
        self._every = every
        self._action_repeat = action_repeat

    def __call__(self, step):
        if self._every is None:
            return False
        every = self._every // self._action_repeat
        if step % every == 0:
            return True
        return False


class Timer:
    def __init__(self):
        self._start_time = time.time()
        self._last_time = time.time()

    def reset(self):
        elapsed_time = time.time() - self._last_time
        self._last_time = time.time()
        total_time = time.time() - self._start_time
        return elapsed_time, total_time

    def total_time(self):
        return time.time() - self._start_time


class TruncatedNormal(pyd.Normal):
    def __init__(self, loc, scale, low=-1.0, high=1.0, eps=1e-6):
        super().__init__(loc, scale, validate_args=False)
        self.low = low
        self.high = high
        self.eps = eps

    def _clamp(self, x):
        clamped_x = torch.clamp(x, self.low + self.eps, self.high - self.eps)
        x = x - x.detach() + clamped_x.detach()
        return x

    def sample(self, clip=None, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        eps = _standard_normal(shape,
                               dtype=self.loc.dtype,
                               device=self.loc.device)
        eps *= self.scale
        if clip is not None:
            eps = torch.clamp(eps, -clip, clip)
        x = self.loc + eps
        return self._clamp(x)


def schedule(schdl, step):
    try:
        return float(schdl)
    except ValueError:
        match = re.match(r'linear\((.+),(.+),(.+)\)', schdl)
        if match:
            init, final, duration = [float(g) for g in match.groups()]
            mix = np.clip(step / duration, 0.0, 1.0)
            return (1.0 - mix) * init + mix * final
        match = re.match(r'step_linear\((.+),(.+),(.+),(.+),(.+)\)', schdl)
        if match:
            init, final1, duration1, final2, duration2 = [
                float(g) for g in match.groups()
            ]
            if step <= duration1:
                mix = np.clip(step / duration1, 0.0, 1.0)
                return (1.0 - mix) * init + mix * final1
            else:
                mix = np.clip((step - duration1) / duration2, 0.0, 1.0)
                return (1.0 - mix) * final1 + mix * final2
    raise NotImplementedError(schdl)

def int2alphanumeric(num):
    assert round(num) == num and num >= 1
    s = ""
    while num:
        s = chars[num % charsLen - 1] + s
        num //= charsLen
    return(s)

def alphanumeric2int(numStr):
  num = 0
  for i, c in enumerate(reversed(numStr)):
    num += (chars.index(c) + 1) * (charsLen ** i)
  return(num)

def get_col(titles, title):
    assert title in titles, 'wrong title %s!' % title
    return int2alphanumeric(titles.index(title) + 1)

def auto_commit(tag):
    repo = pg.Repository('./')
    # if 'logging' not in list(repo.branches):
    #     os.system('git checkout -b logging')
    # else:
    #     os.system('git checkout logging')
    # if repo.diff().stats.files_changed:
    #     print('change detected, input commit message, enter to use tag %s, input n to skip this commit:' % tag)
    #     msg = input()
    #     if not msg == 'n':
    #         os.system('git add .')
    #         # repo.index.add_all()
    #         # author = pg.Signature('Kaizhe Hu', 'hukz18@mails.tsinghua.edu.cn')
    #         os.system("git commit -m '%s'" % (msg or tag))
    return repo.head.target.hex

def create_log(wks, args, cfg, hex, start_time):
    def next_available_row(worksheet):
        str_list = list(filter(None, worksheet.col_values(1)))
        return str(len(str_list) + 1)
    server = libtmux.Server()
    session = server.attached_sessions[0].get('session_name')
    id =  server.attached_sessions[0].attached_window.get('window_index')
    row_pos = next_available_row(wks)
    row = [
        args.config,
        start_time.strftime('%Y/%m/%d %H:%M'),
        '', # end_time
        '', # duration
        hex[:8], # git commit hash
        socket.gethostname(), # machine
        len(os.environ['CUDA_VISIBLE_DEVICES'].split(',')), # GPUs
        args.seeds, # seeds
        ':'.join([session, id]), # Tmux
        '%g' % cfg.num_train_frames, # max_steps
        cfg.batch_size,
        cfg.nstep,
        cfg.frame_stack,
        cfg.action_repeat,
        cfg.feature_dim,
        cfg.agent.stddev_clip,
        args.task,
        str(cfg.aug),
        '', # performance
        '', # solved trajectory
        args.tag
    ]
    wks.insert_row(row, int(row_pos), "user_entered")
    return wks

def end_log(wks, args, res, start_time, end_time):
    titles = list(filter(None, wks.row_values(1)))
    id_list = list(filter(None, wks.col_values(1)))
    row_pos = str(id_list.index(args.config) + 1)
    time_delta = end_time - start_time
    wks.update(get_col(titles, 'Finish (UTC+8)') + row_pos, end_time.strftime('%Y/%m/%d %H:%M'))
    wks.update(get_col(titles, 'Duration') + row_pos, '%dd %dh %dm' % (time_delta.days, time_delta.seconds // 3600, (time_delta.seconds % 3600) // 60 ))
    wks.update(get_col(titles, 'Performance') + row_pos, res.performance)
    wks.update(get_col(titles, 'Solved trajectory') + row_pos, res.solved_traj)