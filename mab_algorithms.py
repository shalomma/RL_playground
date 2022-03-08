import numpy as np
from abc import ABC, abstractmethod

from mab import MAB


class AlgoMAB(ABC):
    def __init__(self, mab: MAB, horizon, eps=0.):
        self.mab = mab
        self.K = len(mab)
        self.T = horizon
        self.eps = eps
        self.means = np.zeros(self.K)
        self.counts = np.ones(self.K)
        self.regret = []
        self.rewards = []
        self.opt_expected_value = np.max(mab.expected_values)
        self.history = []

    @classmethod
    @abstractmethod
    def get_upper_limit_function(cls, k_bandits, horizon):
        pass

    def update_mean(self, k, value):
        a = 1 / self.counts[k]
        self.means[k] = self.means[k] * (1 - a) + value * a

    @staticmethod
    def argmax_random_tie_breaking(a):
        return int(np.random.choice(np.flatnonzero(a == a.max())))

    @abstractmethod
    def select_arm(self, t):
        pass

    def update_arm_history(self, k, reward):
        self.counts[k] += 1
        self.history.append(k)
        self.rewards.append(reward)

    def play_arm(self, k):
        return self.mab.sample(k)

    def learn(self):
        for t in range(1, self.T):
            k = self.select_arm(t)
            reward = self.play_arm(k)
            self.update_mean(k, reward)
            self.update_arm_history(k, reward)
        self.regret = self.opt_expected_value - self.rewards
