import numpy as np

import mab
import plot
from mab_algorithms import AlgoMAB


class UCB1(AlgoMAB):
    def __init__(self, mab, horizon, eps=0.):
        super(UCB1, self).__init__(mab, horizon, eps)

    @classmethod
    def get_upper_limit_function(cls, k_bandits, horizon):
        return np.sqrt(k_bandits * np.arange(1, horizon) * np.log(horizon))

    def radius(self, k, t):
        return np.sqrt(2 * np.log(t) / self.counts[k])

    def select_arm(self, t):
        r = np.array([self.radius(k, t) for k in range(self.K)])
        ucb = self.means + r
        k = self.argmax_random_tie_breaking(ucb) if np.random.uniform(0, 1) > self.eps else np.random.randint(0, self.K)
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
        algo = UCB1(mab_, horizon=T, eps=0.1)
        algo.learn()
        regrets.append(algo.regret)
        histories.append(algo.history)

    plot.plot_expected_regret(regrets)
    plot.plot_expected_history(histories)
