import numpy as np
import gym
from gym import spaces

class stochastic_bandit(gym.Env):
    """
    The stochastic bandit problem.

    Attributes:
    TBC
    """
    def __init__(self, size, reward_distribution = "gaussian"):


        self.k = size
        self.mu = np.zeros(self.k)
        self.action_space = spaces.Discrete(self.k)
        self.observation_space = spaces.Discrete(1)

        # defining a vector (length k) of central values / expected values from a distribution 
        if reward_distribution == "gaussian":
            self.mu = np.random.normal(self.k, 1, size = self.k)
 

    def step(self, action):
        assert self.action_space.contains(action)

        reward = 0
        done = True

        reward = np.random.normal(self.mu[action], 1)
        regret = max(self.mu) - reward

        return reward, regret

    def reset(self):
        return 0

    def render(self, mode='human', close=False):
        pass
