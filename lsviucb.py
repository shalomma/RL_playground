import numpy as np


class LSVIUCB(object):
    def __init__(self, env, K, delta):
        self.env = env
        self.K = K
        self.p = delta
        self.d = self.env.nState * self.env.nAction
        self.lam = 1.0
        self.Lambda = {h: self.lam * np.identity(self.d) for h in range(self.env.epLen)}
        self.L = {h: self.lam * np.identity(self.d) for h in range(self.env.epLen)}
        self.Linv = {h: (1 / self.lam) * np.identity(self.d) for h in range(self.env.epLen)}
        self.w = {h: np.zeros(self.d) for h in range(self.env.epLen)}
        self.Q = {(h, s, a): 0.0 for h in range(self.env.epLen + 1) for s in self.env.states.keys()
                  for a in range(self.env.nAction)}
        self.features_state_action = {(s, a): np.zeros(self.d) for s in self.env.states.keys()
                                      for a in range(self.env.nAction)}
        self.create_identity()
        self.buffer = {h: [] for h in range(self.env.epLen)}
        self.sums = {h: np.zeros(self.d) for h in range(self.env.epLen)}
        self.c = 1.0
        self.m_2 = 3.0  # choosing this is constant is very important, how to do so is not simple though...

    def create_identity(self):
        """
            A function that creates the Identity Matrix for a Dictionary
        """
        i = 0
        for key in self.features_state_action.keys():
            self.features_state_action[key][i] = 1
            i += 1
        # j = 0

    def update_buffer(self, s, a, r, s_, h):
        self.buffer[h].append((s, a, r, s_))

    def reset_buffer(self):
        self.buffer = {h: [] for h in range(self.env.epLen)}

    def update(self):

        Q = {(h, s, a): 0.0 for h in range(self.env.epLen + 1) for s in self.env.states.keys()
             for a in range(self.env.nAction)}
        for h in range(self.env.epLen - 1, -1, -1):
            d = self.buffer[h]
            s, a, r, s_ = d[0][0], d[0][1], d[0][2], d[0][3]

            self.L[h] = self.L[h] + np.outer(self.features_state_action[s, a], self.features_state_action[s, a])

            self.Linv[h] = \
                self.Linv[h] - np.dot((np.outer(np.dot(self.Linv[h], self.features_state_action[s, a]),
                                                self.features_state_action[s, a])), self.Linv[h]) / \
                (1 + np.dot(np.dot(self.features_state_action[s, a], self.Linv[h]),
                            self.features_state_action[s, a]))

            self.sums[h] = self.sums[h] + self.features_state_action[s, a] * (self.env.R[s, a][0] +
                                                                              max(np.array([Q[(h + 1, s_, a)] for a in
                                                                                            range(self.env.nAction)])))

            self.w[h] = np.matmul(self.Linv[h], self.sums[h])
            for ss in self.env.states.keys():
                for aa in range(self.env.nAction):
                    feature = self.features_state_action[ss, aa]
                    Q[h, ss, aa] = min(np.inner(self.w[h], feature) + self.beta(h)
                                       * np.sqrt(np.dot(np.dot(feature, self.Linv[h]), feature)), self.env.epLen)
        self.Q = Q.copy()

    def act(self, s, h):
        """
        A function that returns the argmax of Q given the state and timestep
        """
        return self.env.argmax(np.array([self.Q[(h, s, a)] for a in range(self.env.nAction)]))

    def beta(self, h):
        """
        iota = np.log(2*self.d*self.K*self.env.epLen/self.p)
        return self.c * self.d * (self.env.epLen-h)/2 * np.sqrt(iota)
        """

        first = self.m_2 * np.sqrt(self.lam)
        second = np.sqrt(2 * np.log(1 / self.p) + np.log(np.linalg.det(self.L[h]) / self.lam))
        return first + second

    def run(self):
        R = 0
        Rvec = []
        for _ in (range(1, self.K + 1)):
            self.env.reset()
            done = 0
            while not done:
                s = self.env.state
                h = self.env.timestep
                a = self.act(s, h)
                r, s_, done = self.env.advance(a)
                self.update_buffer(s, a, r, s_, h)
                R += r
            Rvec.append(R)
            self.update()
            self.reset_buffer()
        return Rvec
