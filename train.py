# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from datetime import datetime, timedelta
import sys
import warnings
import ipdb
from easydict import EasyDict
from numpy.lib.stride_tricks import DummyArray
warnings.filterwarnings('ignore', category=DeprecationWarning)

import os
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'

from pathlib import Path

import hydra
import numpy as np
import torch
import argparse
import gspread
from dm_env import specs

import dmc
import utils
from logger import Logger, DummyLogger
from simple_replay_buffer import ReplayBuffer, make_replay_loader
from video import TrainVideoRecorder, VideoRecorder
import gtimer as gt
from drqv2 import DrQV2Agent
global args
torch.backends.cudnn.benchmark = True


def make_agent(obs_spec, action_spec, aug, cfg):
    cfg.obs_shape = obs_spec.shape
    cfg.action_shape = action_spec.shape
    cfg.aug = aug
    # return hydra.utils.instantiate(cfg)
    # from drqv2 import DrQV2Agent
    cfg = dict(cfg)
    cfg.pop('_target_')
    return DrQV2Agent(**cfg)


class Workspace:
    def __init__(self, cfg, args):
        self.work_dir = Path.cwd()
        print(f'workspace: {self.work_dir}')

        self.cfg = cfg
        self.args = args
        utils.set_seed_everywhere(cfg.seed)
        # self.device = torch.device(cfg.device)
        self.setup()

        self.agent = make_agent(self.train_env.observation_spec(),
                                self.train_env.action_spec(), 
                                self.cfg.aug, self.cfg.agent)
        self.timer = utils.Timer()
        self._global_step = 0
        self._global_episode = 0

    def setup(self):
        # create logger
        self.logger = Logger(self.work_dir, use_tb=self.cfg.use_tb) if not self.args.debug else DummyLogger()
        # create envs
        self.train_env = dmc.make(self.cfg.task_name, self.cfg.frame_stack,
                                  self.cfg.action_repeat, self.cfg.seed)
        self.eval_env = dmc.make(self.cfg.task_name, self.cfg.frame_stack,
                                 self.cfg.action_repeat, self.cfg.seed)
        # create replay buffer
        data_specs = (self.train_env.observation_spec(),
                      self.train_env.action_spec(),
                      specs.Array((1,), np.float32, 'reward'),
                      specs.Array((1,), np.float32, 'discount'))

        self.replay_buffer = ReplayBuffer(data_specs, self.cfg.replay_buffer_size, 
            self.cfg.nstep, self.cfg.discount, self.cfg.batch_size)
        self.replay_loader = make_replay_loader(
            self.replay_buffer, self.cfg.batch_size, self.cfg.replay_buffer_num_workers)
        self._replay_iter = None

        self.video_recorder = VideoRecorder(
            self.work_dir if (self.cfg.save_video and not self.args.debug) else None)
        self.train_video_recorder = TrainVideoRecorder(
            self.work_dir if (self.cfg.save_train_video and not self.args.debug) else None)


    @property
    def global_step(self):
        return self._global_step

    @property
    def global_episode(self):
        return self._global_episode

    @property
    def global_frame(self):
        return self.global_step * self.cfg.action_repeat

    @property
    def replay_iter(self):
        if self._replay_iter is None:
            self._replay_iter = iter(self.replay_loader)
        return self._replay_iter

    def eval(self):
        step, episode, rewards = 0, 0, []
        eval_until_episode = utils.Until(self.cfg.num_eval_episodes)

        while eval_until_episode(episode):
            total_reward = 0
            time_step = self.eval_env.reset()
            self.video_recorder.init(self.eval_env, enabled=(episode == 0))
            while not time_step.last():
                with torch.no_grad(), utils.eval_mode(self.agent): # set all model to eval then restore origin state
                    action = self.agent.act(time_step.observation,
                                            self.global_step,
                                            eval_mode=True)
                time_step = self.eval_env.step(action)
                self.video_recorder.record(self.eval_env)
                total_reward += time_step.reward
                step += 1

            episode += 1
            rewards.append(total_reward)
            self.video_recorder.save(f'{self.global_frame}.mp4')

        with self.logger.log_and_dump_ctx(self.global_frame, ty='eval') as log:
            log('episode_reward', np.mean(rewards))
            log('episode_length', step * self.cfg.action_repeat / episode)
            log('episode', self.global_episode)
            log('step', self.global_step)
        return np.mean(rewards), np.std(rewards)

    @gt.wrap
    def train(self):
        # predicates
        train_until_step = utils.Until(self.cfg.num_train_frames, self.cfg.action_repeat)
        seed_until_step = utils.Until(self.cfg.num_seed_frames, self.cfg.action_repeat)
        eval_every_step = utils.Every(self.cfg.eval_every_frames, self.cfg.action_repeat)

        episode_step, episode_reward = 0, 0
        time_step = self.train_env.reset()
        self.replay_buffer.add(time_step) # type: first step/midium step/last step, obs, act, rew
        self.train_video_recorder.init(time_step.observation)
        metrics = None
        gt.stamp('start training')
        train_loop = gt.timed_loop('train_loop')
        while train_until_step(self.global_step):
            next(train_loop)
            if time_step.last(): # episode ends
                # ipdb.set_trace()
                self._global_episode += 1
                self.train_video_recorder.save(f'{self.global_frame}.mp4')
                # wait until all the metrics schema is populated
                if metrics is not None:
                    # log stats
                    elapsed_time, total_time = self.timer.reset()
                    episode_frame = episode_step * self.cfg.action_repeat
                    with self.logger.log_and_dump_ctx(self.global_frame,
                                                      ty='train') as log:
                        log('fps', episode_frame / elapsed_time)
                        log('total_time', total_time)
                        log('episode_reward', episode_reward)
                        log('episode_length', episode_frame)
                        log('episode', self.global_episode)
                        log('buffer_size', len(self.replay_buffer))
                        log('step', self.global_step)

                # reset env
                time_step = self.train_env.reset()
                
                self.replay_buffer.add(time_step)
                self.train_video_recorder.init(time_step.observation)
                # try to save snapshot
                # if self.cfg.save_snapshot:
                #     self.save_snapshot()
                if episode_reward > self.cfg.threshold:
                    break
                episode_step = 0
                episode_reward = 0
                gt.stamp('reset env', unique=False)

            # evaluate every cfg.eval_every_frames frame(50000)
            if eval_every_step(self.global_step):
                self.logger.log('eval_total_time', self.timer.total_time(),
                                self.global_frame)
                self.eval()
                gt.stamp('eval', unique=False)

            # sample action
            with torch.no_grad(), utils.eval_mode(self.agent):
                action = self.agent.act(time_step.observation,
                                        self.global_step,
                                        eval_mode=False)
                gt.stamp('get action', unique=False)

            # try to update the agent after origin frames
            if not seed_until_step(self.global_step):
                metrics = self.agent.update(self.replay_iter, self.global_step)
                self.logger.log_metrics(metrics, self.global_frame, ty='train')
                gt.stamp('update', unique=False)

            # take env step
            time_step = self.train_env.step(action)
            episode_reward += time_step.reward
            self.replay_buffer.add(time_step)
            gt.stamp('add timestep', unique=False)
            self.train_video_recorder.record(time_step.observation)
            episode_step += 1
            self._global_step += 1
        train_loop.exit()
    def save_snapshot(self):
        snapshot = self.work_dir / 'snapshot.pt'
        keys_to_save = ['agent', 'timer', '_global_step', '_global_episode']
        payload = {k: self.__dict__[k] for k in keys_to_save}
        with snapshot.open('wb') as f:
            torch.save(payload, f)

    def load_snapshot(self):
        snapshot = self.work_dir / 'snapshot.pt'
        with snapshot.open('rb') as f:
            payload = torch.load(f)
        for k, v in payload.items():
            self.__dict__[k] = v


from train import Workspace as W
@hydra.main(config_path='cfgs', config_name='config')
def main(cfg):
    root_dir = Path.cwd()
    gc = gspread.service_account()
    wks = gc.open("GP - Kaizhe").worksheet('DrQ_v2')
    # start_time = datetime.utcnow() + timedelta(hours=8)
    start_time = datetime.now()
    seeds = np.random.choice(0xff, args.seeds, replace=False).tolist()
    if not args.debug:
        hex = utils.auto_commit(args.tag)
        utils.create_log(wks, args, cfg, hex, start_time) # TODO: modify create log here
    avg_best, std_of_best = -np.Infinity, 0
    solved_traj = []
    for seed in gt.timed_for(seeds):
        cfg.seed = seed
        workspace = W(cfg, args)
        gt.stamp('init workspace')
        # if snapshot.exists():
        #     print(f'resuming: {snapshot}')
        #     workspace.load_snapshot()
        workspace.train()
        gt.stamp('training')
        solved_traj.append(workspace.global_episode)
        workspace.logger.log('eval_total_time', workspace.timer.total_time(), workspace.global_frame)
        mean, std = workspace.eval()
        gt.stamp('eval')
        if mean > avg_best:
            avg_best, std_of_best = mean, std
            workspace.save_snapshot()
            gt.stamp('save snapshot')
    res = EasyDict({
        'performance': '%.1f±%.1f' % (avg_best, std_of_best),
        'solved_traj': np.mean(solved_traj)
    })
    # end_time = datetime.utcnow() + timedelta(hours=8)
    end_time = datetime.now()
    print('used seeds:', seeds)
    if not args.debug:
        utils.end_log(wks, args, res, start_time, end_time)
    with open('gtimer.log', 'w') as f:
        f.write(gt.report(include_itrs=False, format_options = {'stamp_name_width': 30, 'itr_num_width': 10}))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', type=int, default=1, help='num of repeat experiments')
    parser.add_argument('--debug', action='store_true', help='whether to log weights')
    parser.add_argument('--config', help='specify config file')
    parser.add_argument('--job', type=int, help='specify job id, config will increase based on id')
    parser.add_argument('--tag', type=str, help='brief description for the purpose of the run')
    parser.add_argument('--task', type=str, help='specify task for dmc')
    args, unknown = parser.parse_known_args()

    sys.argv = ['run.py'] + sys.argv[sys.argv.index('--hydra') + 1:] if '--hydra' in sys.argv else ['run.py']
    sys.argv.append('task=%s' % args.task)
    main()