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
        if self.state == 0 and action == 0:
            reward = 0
            done = True
        elif self.state == 0 and action == 1:
            self.state = 1
            reward = 0
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
    env = MaxBiasEnv(mean=-0.0001, var=1.0)

    n_epochs = 1000
    action_epochs_history = []
    for _ in range(n_epochs):
        learner = QLearning(env, lr=0.1, gamma=1, eps=0.1)
        n_episodes = 350
        state_ = env.state
        action_history = []
        while n_episodes > 0:
            state_, reward_, action_, done_ = learner.predict(state_)
            if done_:
                action_history.append(state_)
                env.reset()
                state_ = env.state
                n_episodes -= 1
        action_epochs_history.append(action_history)

    plt.figure(figsize=(12, 8))
    plt.plot(np.array(action_epochs_history).mean(axis=0))
    plt.xlabel('episode')
    plt.ylabel('% left actions')
    plt.show()
