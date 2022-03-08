import numpy as np
import matplotlib.pyplot as plt


def plot_expected_regret(regrets):
    plt.figure(figsize=(12, 8))
    # upper_limit = UCB1.get_upper_limit_function(k_bandits=len(mab_), horizon=T)
    expected_regret = np.array(regrets).mean(axis=0)
    expected_accumulated_regret = np.cumsum(expected_regret)
    plt.plot(expected_accumulated_regret)
    # plt.plot(upper_limit)
    plt.xlabel('t')
    plt.ylabel('E[regret]')
    plt.show()


def plot_expected_history(histories):
    plt.figure(figsize=(12, 8))
    expected_history = np.array(histories).mean(axis=0)
    plt.plot(expected_history)
    plt.xlabel('t')
    plt.yticks(np.arange(0, 1., 0.05))
    plt.show()
