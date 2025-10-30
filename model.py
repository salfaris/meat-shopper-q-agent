import numpy as np


class MeatBuyingQLAgent:
    def __init__(
        self,
        N: int,
        num_actions: int,
        alpha: float = 0.1,
        gamma: float = 0.95,
        epsilon: float = 0.2,
    ):
        self.N = N
        self.num_actions = num_actions

        # Q-table shape = (num states approx) x (num actions ^ N)
        self.q_table = np.zeros((num_actions,) * N)

        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def select_action(self, rng: np.random.Generator):
        proba = np.random.rand() < self.epsilon
        if proba:
            return rng.integers(self.num_actions, size=self.N)
        else:
            idx = np.unravel_index(
                np.argmax(self.q_table, axis=None), self.q_table.shape
            )
            return np.array(idx)

    def update(self, action, reward):
        """Q-learning update on single-step episode."""
        idx = tuple(action)
        old_value = self.q_table[idx]
        new_value = (1 - self.alpha) * old_value + (self.alpha * reward)
        self.q_table[idx] = new_value
