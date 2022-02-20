from gym import Env
from gym.spaces import Discrete
from q_learning import QLearning


class LineEnv(Env):
    def __init__(self, n):
        assert n > 0 and isinstance(n, int)
        self.n = n
        self.state_space = Discrete(n)
        self.action_space = Discrete(1)
        self.state = 0

    def step(self, action):
        self.state += 1
        done = self.state == (self.n - 1)
        reward = 1 if done else 0
        return self.state, reward, done, {}

    def reset(self):
        self.state = 0

    def render(self, mode="human"):
        pass


if __name__ == '__main__':
    env = LineEnv(n=1000)
    learner = QLearning(env, lr=.1, gamma=1, eps=0)
    eps = 1e-2
    n_episodes = 1
    state_ = env.state
    while learner.q_table[0][0] <= 1 - eps:
        # learner.lr = 1 / n_episodes
        state_, _, _, done_ = learner.predict(state_)
        if done_:
            env.reset()
            state_ = env.state
            n_episodes += 1
