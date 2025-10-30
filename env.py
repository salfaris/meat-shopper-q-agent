import numpy as np

MAX_WEIGHT = 2000
MAX_DAYS_TO_EXPIRY = 10


class MeatBuyingDiscreteEnv:
    def __init__(
        self,
        rng: np.random.Generator,
        N: int = 3,
        budget: int = 100,
        required_grams: int = 2000,
        expiry_threshold=3,
    ):
        self.N = N  # Number of meat_types
        self.budget = budget
        self.required_grams = required_grams
        self.expiry_threshold = expiry_threshold
        self.weight_options = [
            0,
            250,
            500,
            1000,
            MAX_WEIGHT,
        ]  # predefined weights in grams
        self.num_actions = len(self.weight_options)
        self.reset(rng)

    def price_per_gram(self, weight: float):
        """Price decreases with weight."""
        if weight == 0:
            return 0
        base_price = 0.01 * (
            MAX_WEIGHT / weight
        )  # Fixed currently, but this will be dynamic!
        return base_price

    def reset(self, rng: np.random.Generator):
        """Initialize available meat attributes for the environment."""
        self.is_halal = rng.choice([0, 1], self.N, p=[0.6, 0.4])
        self.is_discounted = rng.choice([0, 1], self.N, p=[0.97, 0.03])
        self.days_to_expiry = rng.integers(1, MAX_DAYS_TO_EXPIRY, self.N)
        self.total_bought = 0
        self.total_cost = 0

        # For simplicity, just track current indices because attributes are fixed for
        # each episode.
        self.state = 0

        return self._get_obs()

    def _get_obs(self):
        """Return state as vector of current meat attributes.

        Static because there is no dynamic state usage. So obs is always the same.
        But we put it here to conform with standard RL practices.
        """
        return np.concatenate(
            (
                self.is_halal,
                self.is_discounted,
                self.days_to_expiry / MAX_DAYS_TO_EXPIRY,  # normalized for scale
            )
        )

    def step(self, actions):
        weights = np.array([self.weight_options[a] for a in actions])
        price_per_g = np.array([self.price_per_gram(w) for w in weights])
        costs = weights * price_per_g

        total_weight = weights.sum()
        total_cost = costs.sum()

        # Reward calculation
        reward = 0
        reward -= total_cost / self.budget  # minimize cost!
        reward += np.sum(weights * self.is_discounted) / 1000  # Reward discounted
        reward -= (
            np.sum(weights * self.days_to_expiry < self.expiry_threshold) / 500
        )  # Penalize on choosing close to expiry meat.

        # Might want to tune this factor later?
        non_halal_penalty_factor = 0.1
        excess_weight_penalty_factor = 0.001

        # Heavy penalty for non-halal
        non_halal_weights = weights * (1 - self.is_halal)
        reward -= np.sum(non_halal_weights**2) * non_halal_penalty_factor

        # Heavy penalty for exceeding required grams (overbuying)
        excess = max(0, np.sum(weights) - self.required_grams)
        reward -= (excess**2) * excess_weight_penalty_factor

        self.total_bought += total_weight
        self.total_cost += total_cost

        done = (self.total_bought >= self.required_grams) or (
            self.total_cost > self.budget
        )

        return self._get_obs(), reward, done, {}
