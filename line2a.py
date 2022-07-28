from gym import Env
from gym.spaces import Discrete
from q_learning import QLearning
import numpy as np
import matplotlib.pyplot as plt


class Line2ActionEnv(Env):
    def __init__(self, n, start, horizon):
        self.n = n
        self.state_space = Discrete(n)
        self.action_space = Discrete(2)
        self.start = start
        self.horizon = horizon

        self.state = None
        self.count = None
        self.reset()

    def step(self, action):
        self.state += -1 if action == 0 else 1
        self.state = int(np.clip(self.state, 0, self.n - 1))
        self.count += 1
        done = self.count == self.horizon
        reward = 1 if self.state in [self.n - 1, 0] else 0
        return self.state, reward, done, {}

    def reset(self):
        self.state = self.start
        self.count = 0

    def render(self, mode="human"):
        pass


if __name__ == '__main__':
    env = Line2ActionEnv(n=50, start=40, horizon=100)

    visitation = [0] * env.n

    n_episodes = 100
    done_ = False
    for _ in range(n_episodes):
        learner = QLearning(env, lr=0.1, gamma=0.1, eps=0.1)
        state_ = env.state
        while not done_:
            state_, reward_, _, done_ = learner.predict(state_)
            visitation[state_] += 1
        done_ = False
        env.reset()

    plt.figure(figsize=(15, 6))
    plt.plot(np.array(visitation) / n_episodes)
    plt.xlabel('states')
    plt.show()
