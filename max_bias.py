import numpy as np
from gym import Env
from gym.spaces import Discrete
from q_learning import QLearning
import matplotlib.pyplot as plt


class MaxBiasEnv(Env):
    def __init__(self, mean, var):
        self.mean = mean
        self.var = var
        self.state_space = Discrete(2)
        self.action_space = Discrete(2)
        self.state = 0

    def step(self, action):
        if self.state == 0 and action == 0:  # right
            reward = 0.
            done = True
        elif self.state == 0 and action == 1:  # left
            self.state = 1
            reward = 0.
            done = False
        else:  # state == 1
            reward = np.random.normal(loc=self.mean, scale=self.var)
            done = True
        return self.state, reward, done, {}

    def reset(self):
        self.state = 0

    def render(self, mode="human"):
        pass


if __name__ == '__main__':
    experiments = {
        '-0.1': -0.1,
        '0.': 0.,
        '0.1': 0.1,
        '0.2': 0.2,
        '0.3': 0.3,
        '0.4': 0.4,
    }

    env = MaxBiasEnv(mean=-0.1, var=1.0)
    learner = QLearning(env, lr=0.1, gamma=1., eps=0.)

    n_epochs = 100
    n_episodes = 350
    action_epochs_history = []
    for _ in range(n_epochs):
        state_ = env.state
        action_history = []
        for i in range(1, n_episodes):
            done_ = False
            while not done_:
                state_, reward_, action_, done_ = learner.predict(state_)
            action_history.append(state_)
            env.reset()
            state_ = env.state
        learner.reset()
        action_epochs_history.append(action_history)

    plt.figure(figsize=(12, 8))
    plt.plot(np.array(action_epochs_history).mean(axis=0))
    plt.xlabel('episode')
    plt.ylabel('% left actions')
    plt.yticks(np.arange(0, 1, 0.05))
    plt.show()
