import torch
import random
from collections import defaultdict
import numpy as np
from torch.utils.data import IterableDataset


def episode_len(episode):
    # subtract -1 because the dummy first transition
    return next(iter(episode.values())).shape[0] - 1


class ReplayBuffer(IterableDataset):
    def __init__(self, data_specs, max_size, nstep, discount, fetch_every=1000) -> None:
        self._data_specs = data_specs
        self._current_episode = defaultdict(list)
        self._num_episodes = 0
        self._num_transitions = 0
        self._episodes = {}
        self._episode_fns = []
        self._replay_fns = []
        self._replay_episodes = {}

        self._size = 0
        self._max_size = max_size
        self._nstep = nstep
        self._discount = discount
        self._fetch_every = fetch_every
        self._samples_since_last_fetch = fetch_every

    def __len__(self):
        return self._num_transitions

    def add(self, time_step):
        for spec in self._data_specs:
            value = time_step[spec.name]
            if np.isscalar(value):
                value = np.full(spec.shape, value, spec.dtype)
            assert spec.shape == value.shape and spec.dtype == value.dtype
            self._current_episode[spec.name].append(value)
        if time_step.last():
            episode = dict()
            for spec in self._data_specs:
                value = self._current_episode[spec.name]
                episode[spec.name] = np.array(value, spec.dtype)
            self._current_episode = defaultdict(list)
            self._store_episode(episode)

    def _store_episode(self, episode):
        eps_idx = self._num_episodes
        eps_len = episode_len(episode)
        eps_fn = f'{eps_idx}_{eps_len}'
        self._replay_fns.append(eps_fn)
        self._replay_episodes[eps_fn] = episode

    def _fetch_episode(self, eps_fn):
        episode = self._replay_episodes[eps_fn]
        eps_len = episode_len(episode)
        while eps_len + self._size > self._max_size:
            early_eps_fn = self._episode_fns.pop(0)
            early_eps = self._episodes.pop(early_eps_fn)
            self._num_episodes -= 1
            self._size -= episode_len(early_eps)
        self._episode_fns.append(eps_fn)
        self._episode_fns.sort()  # TODO: why sort?
        self._episodes[eps_fn] = episode
        self._size += eps_len
        self._replay_episodes.pop(eps_fn)
        self._num_episodes += 1

    def _sample_episode(self):
        eps_fn = random.choice(self._episode_fns)
        return self._episodes[eps_fn]

    def _try_fetch(self):
        if self._samples_since_last_fetch < self._fetch_every:
            return
        self._samples_since_last_fetch = 0
        for _ in range(len(self._replay_fns)):
            eps_fn = self._replay_fns.pop(0)
            if eps_fn in self._episodes.keys():
                break
            self._fetch_episode(eps_fn)

    def _sample(self):
        self._try_fetch()
        self._samples_since_last_fetch += 1
        episode = self._sample_episode()
        # add +1 for the first dummy transition
        idx = np.random.randint(0, episode_len(episode) - self._nstep + 1) + 1
        obs = episode['observation'][idx - 1]
        action = episode['action'][idx]
        next_obs = episode['observation'][idx + self._nstep - 1]
        reward = np.zeros_like(episode['reward'][idx])
        discount = np.ones_like(episode['discount'][idx])
        for i in range(self._nstep):
            step_reward = episode['reward'][idx + i]
            reward += discount * step_reward
            discount *= episode['discount'][idx + i] * self._discount
        return (obs, action, reward, discount, next_obs)

    def __iter__(self):
        while True:
            yield self._sample()


def make_replay_loader(iterable, batch_size, num_workers):
    loader = torch.utils.data.DataLoader(iterable,
                                         batch_size=batch_size,
                                         num_workers=num_workers,
                                         pin_memory=True)
    return loader