import random
import numpy as np


class QLearning:
    def __init__(self, env, lr=1e-1, gamma=6e-1, eps=1e-1):
        self.env = env
        self.eps = eps
        self.lr = lr
        self.gamma = gamma
        self.q_table = np.zeros((env.state_space.n, env.action_space.n))
        # self.q_table = np.random.rand(env.state_space.n, env.action_space.n)

    def get_action(self, state):
        return np.argmax(self.q_table[state]) if random.uniform(0, 1) > self.eps else self.env.action_space.sample()

    def predict(self, state):
        action = self.get_action(state)
        next_state, reward, done, info = self.env.step(action)

        old_value = self.q_table[state, action]
        next_max = np.max(self.q_table[next_state])

        new_value = (1 - self.lr) * old_value + self.lr * (reward + self.gamma * next_max)
        self.q_table[state, action] = new_value

        return next_state, reward, action, done
