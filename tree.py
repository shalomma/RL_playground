import numpy as np
from gym import Env
from gym.spaces import Discrete
from q_learning import QLearning
import matplotlib.pyplot as plt


class TreeEnv(Env):
    def __init__(self, mean, variance):
        self.mean = mean
        self.variance = variance
        self.action_space = Discrete(2)
        self.state_space = Discrete(7)
        self.tree = [[0], [1, 2], [3, 4, 5, 6]]
        self.state = 0
        self.depth = 0

    def step(self, action):
        self.depth += 1
        done = False
        if self.depth == 1:
            self.state = 1 if action == 0 else 2
        elif self.depth == 2:
            done = True
            if self.state == 1:
                self.state = 3 if action == 0 else 4
            elif self.state == 2:
                self.state = 5 if action == 0 else 6
            else:
                raise Exception('unknown state in depth 1')

        reward = 0 if self.state == 6 else np.random.normal(loc=self.mean, scale=self.variance)
        return self.state, reward, done, {}

    def reset(self):
        self.state = 0
        self.depth = 0

    def render(self, mode="human"):
        pass


if __name__ == '__main__':
    env = TreeEnv(mean=-0.1, variance=1.0)

    n_epochs = 200
    n_episodes = 3000
    action_epochs_history = []
    for _ in range(n_epochs):
        learner = QLearning(env, lr=0.1, gamma=1, eps=0.1)
        episode = 1
        state_ = env.state
        max_reward_visits = []
        while n_episodes > episode:
            state_, reward_, action_, done_ = learner.predict(state_)
            if done_:
                max_reward_visits += [1] if state_ == 6 else [0]
                env.reset()
                state_ = env.state
                episode += 1
        action_epochs_history.append(max_reward_visits)

    plt.figure(figsize=(12, 8))
    plt.plot(np.array(action_epochs_history).mean(axis=0))
    plt.xlabel('episode')
    plt.ylabel('% getting to state 6')
    plt.yticks(np.arange(0, 1.01, 0.05))
    plt.show()
