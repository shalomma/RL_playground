from gym import Env
from gym.spaces import Discrete
import random
import numpy as np
import matplotlib.pyplot as plt
from q_learning import QLearning


class WindowEnv(Env):
    def __init__(self, length, window: tuple, start: tuple, episode_length):
        self.action_space = Discrete(3)
        self.state_space = Discrete(length)
        self.length = length
        self.episode_length = episode_length
        self.window = window
        self.start = start

        self.counter = 0
        self.state = self.init_state()

    def init_state(self):
        return random.randint(self.start[0], self.start[1])

    def step(self, action: int):
        reward = self.get_reward()
        self.state = np.clip(self.state + action - 1, 0, self.length - 1)
        done = self.is_done()
        return self.state, reward, done, {}

    def get_reward(self):
        return 0 if self.window[0] <= self.state <= self.window[1] else -1

    def is_done(self):
        self.counter += 1
        return True if self.episode_length <= self.counter else False

    def reset(self):
        self.counter = 0
        self.state = self.init_state()

    def render(self, mode="human"):
        pass


if __name__ == '__main__':
    env = WindowEnv(length=60, window=(20, 40), start=(0, 1), episode_length=50)
    learner = QLearning(env, lr=.1, gamma=0.9, eps=1e-1)
    n_episodes = 3000

    state_ = env.state
    mean_episode_state_history = []
    episode_state_history = []
    episode_reward = 0
    rewards = []
    while True:
        state_, reward_, _, done_ = learner.predict(state_)
        if done_:
            env.reset()
            state_ = env.state
            mean_episode_state_history.append(np.array(episode_state_history).mean())
            episode_state_history = []
            rewards.append(episode_reward)
            episode_reward = 0
            n_episodes -= 1
            if n_episodes == 0:
                break
        episode_reward += reward_
        episode_state_history.append(state_)

    plt.figure(figsize=(15, 6))
    plt.plot(mean_episode_state_history, label='avg_state')
    plt.plot(rewards, label='total_reward')
    plt.xlabel('episode')
    plt.legend()
    plt.show()
