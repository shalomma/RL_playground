import numpy as np

import mab
import plot
from mab_algorithms import AlgoMAB


class EpsilonGreedy(AlgoMAB):
    def __init__(self, mab, horizon, eps):
        super(EpsilonGreedy, self).__init__(mab, horizon, eps)
        # self.means = np.array([0.0, 0.0001])

    @classmethod
    def get_upper_limit_function(cls, k_bandits, horizon):
        pass

    def select_arm(self, t):
        k = self.argmax_random_tie_breaking(self.means) \
            if np.random.uniform(0, 1) > self.eps else np.random.randint(0, self.K)
        return k


if __name__ == '__main__':
    T = 500
    trials = 1000
    expected_values_ = [0.6, 0.5]
    std_ = [1., 1.]

    mab_ = mab.GaussianMAB(expected_values_, std_)
    regrets = []
    histories = []
    for _ in range(trials):
        algo = EpsilonGreedy(mab_, horizon=T, eps=0.1)
        algo.learn()
        regrets.append(algo.regret)
        histories.append(algo.history)

    plot.plot_expected_regret(regrets)
    plot.plot_expected_history(histories)
