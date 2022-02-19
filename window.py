from gym import Env
from gym.spaces import Discrete
import random
import numpy as np
import matplotlib.pyplot as plt
from q_learning import QLearning


class WindowEnv(Env):
    def __init__(self, length, window_size, episode_length, init_offset=0, init_random_range=0):
        assert window_size + abs(init_offset) + init_random_range < length
        assert window_size > 0 and init_random_range >= 0, episode_length > 0
        self.action_space = Discrete(3)
        self.state_space = Discrete(length)
        self.length = length
        self.episode_length = episode_length
        self.init_offset = init_offset
        self.init_random_range = init_random_range
        self.i_window_start = length // 2 - window_size // 2
        self.i_window_end = self.i_window_start + window_size
        self.counter = 0
        self.state = self.init_state()

    def init_state(self):
        center = self.length // 2 + self.init_offset
        return random.randint(center - self.init_random_range // 2, center + self.init_random_range // 2)

    def step(self, action: int):
        reward = self.get_reward()
        self.state = np.clip(self.state + action - 1, 0, self.length - 1)
        done = self.is_done()
        return self.state, reward, done, {}

    def get_reward(self):
        return 0 if self.i_window_start <= self.state <= self.i_window_end else -1

    def is_done(self):
        self.counter += 1
        return True if self.episode_length <= self.counter else False

    def reset(self):
        self.counter = 0
        self.state = self.init_state()

    def render(self, mode="human"):
        pass


if __name__ == '__main__':
    env = WindowEnv(length=60, window_size=10, episode_length=50, init_offset=10, init_random_range=0)
    learner = QLearning(env, lr=.1, gamma=0.9, eps=4e-2)
    n_episodes = 3000

    state_ = env.state
    mean_episode_state_history = []
    episode_state_history = []
    episode_reward = 0
    rewards = []
    while True:
        state_, reward_, done_ = learner.predict(state_)
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

    plt.figure(figsize=(20, 8))
    plt.plot(mean_episode_state_history)
    # plt.show()
    plt.plot(rewards)
    plt.show()
