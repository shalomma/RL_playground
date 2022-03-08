import numpy as np
from abc import ABC, abstractmethod


class MAB(ABC):
    def __init__(self, expected_values):
        self.expected_values = expected_values

    @abstractmethod
    def sample(self, k):
        raise NotImplemented

    def __len__(self):
        return len(self.expected_values)


class UniformMAB(MAB):
    def __init__(self, expected_values, margin):
        super(UniformMAB, self).__init__(expected_values)
        self.margin = margin

    def sample(self, k):
        return np.random.uniform(low=self.expected_values[k] - self.margin,
                                 high=self.expected_values[k] + self.margin)


class GaussianMAB(MAB):
    def __init__(self, expected_values, std):
        super(GaussianMAB, self).__init__(expected_values)
        self.std = std

    def sample(self, k):
        return np.random.normal(loc=self.expected_values[k], scale=self.std[k])


class BernoulliMAB(MAB):
    def __init__(self, expected_values):
        super().__init__(expected_values)

    def sample(self, k):
        return np.random.binomial(1, self.expected_values[k])
