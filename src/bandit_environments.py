import numpy as np
import gym
from gym import spaces
from src.utilities import * 








##################################################################################################
##################################### -- STOCHASTIC BANDITS -- ###################################
##################################################################################################
class stochastic_bandit(base_bandit_env):
    """
    The stochastic bandit problem as OpenAI Gymnasium environment. 

    Attributes:
    ----------
    size: Number of bandit arms 
    reward_distribution: Theoretical distribution of rewards. Must be one of ["gaussian","lognormal"]
    stochastic_moments: First and second moments defining how expected rewards are distributed. 
    """
    def __init__(self, size, reward_distribution = "gaussian", stochastic_moments = [0,1]):


        reward_distribution_options = ["gaussian", "lognormal"]
        if reward_distribution not in reward_distribution_options:
            raise ValueError("reward distribution must be one of: {}".format(reward_distribution_options))
        self.reward_distribution = reward_distribution


        if not isinstance(stochastic_moments, (list, np.ndarray)):
            raise ValueError("stochastic moments must be one of the following Types [list, np.ndarray]")

        if len(stochastic_moments) != 2:
            raise ValueError("input for stochastic moments must be of lenght 2 (covering first and second moment)")

        # initialising stochastic moments
        self.mu = np.random.normal(stochastic_moments[0], stochastic_moments[1], size=size) # central moments for rewards follow Norm(mu, sigma)

        # init from base environment class
        super().__init__(size)
        
            
    def step(self, action):
        assert self.action_space.contains(action)

        reward = 0
        done = True

        if self.reward_distribution == "gaussian":
            reward = np.random.normal(self.mu[action], 1)

        if self.reward_distribution == "lognormal":
            reward = np.random.lognormal(mean = self.mu[action], sigma = 1)

        regret = max(self.mu) - reward

        return reward, regret







class stochastic_bandit_custom(base_bandit_env):
    """
    Customisable stochastic bandit. Gives more flexibility compared to standard stochastic bandit environment. 

    Attributes:
    ----------
    size: Number of bandit arms 
    reward_distribution: Theoretical distribution of rewards. Must be one of ["gaussian","lognormal"]
    mu: A list of expected values for reward distributions. Type -> [list]
    sigma: A list of sigma values, defining the scale parameters for reward distributions. Type -> [list]
    """
    def __init__(self, size, mu, sigma, reward_distribution = "gaussian"):


        reward_distribution_options = ["gaussian", "lognormal"]
        if reward_distribution not in reward_distribution_options:
            raise ValueError("reward distribution must be one of: {}".format(reward_distribution_options))
        self.reward_distribution = reward_distribution


        if not isinstance(mu, list) or len(mu) != size:
            raise ValueError("mu must be of type *list* and of length *size*")

        if not isinstance(sigma, list) or len(sigma) != size:
            raise ValueError("sigma must be of type *list* and of length *size*")

        
        self.mu = np.array(mu)
        self.sigma = np.array(sigma)

        # init from base environment class
        super().__init__(size)
        
            
    def step(self, action):
        assert self.action_space.contains(action)

        reward = 0
        done = True

        if self.reward_distribution == "gaussian":
            reward = np.random.normal(self.mu[action], self.sigma[action]) 

        if self.reward_distribution == "lognormal":
            reward = np.random.lognormal(mean = self.mu[action], sigma = self.sigma[action])

        regret = max(self.mu) - reward

        return reward, regret





##################################################################################################
##################################### -- ADVERSARIAL BANDITS -- ##################################
##################################################################################################
class adversarial_bandit_rw(base_bandit_env):
    """
    Aversarial bandit environment, where rewards follow a random walk. 

    Attributes:
    ----------
    size: Number of bandit arms 
    reward_distribution: Theoretical distribution of rewards. Must be one of ["gaussian","lognormal"]
    mu: Defines the central moment of the reward distribution. 
        Either "default" or a an np.array of length *size*.
    sigma: Defines the second moment of the reward distribution. Type -> int, float, complex
    """
    def __init__(self, size, reward_distribution = "gaussian", stochastic_moments = [0,1]):


        reward_distribution_options = ["gaussian", "lognormal"]
        if reward_distribution not in reward_distribution_options:
            raise ValueError("reward distribution must be one of: {}".format(reward_distribution_options))
        self.reward_distribution = reward_distribution


        if not isinstance(stochastic_moments, (list, np.ndarray)):
            raise ValueError("stochastic moments must be one of the following Types [list, np.ndarray]")

        if len(stochastic_moments) != 2:
            raise ValueError("input for stochastic moments must be of lenght 2 (covering first and second moment)")

        # initialising stochastic moments
        self.mu = np.random.normal(stochastic_moments[0], stochastic_moments[1], size=size)
       
        # increment for random walks. 
        self.rw_increment = np.zeros(size) 
        # init from base environment class
        super().__init__(size)
        
            
    def step(self, action):
        assert self.action_space.contains(action)

        reward = 0
        done = True

        # new 
        if self.reward_distribution == "gaussian":
            reward = np.random.normal(self.mu[action], 1) + self.rw_increment[action]

        if self.reward_distribution == "lognormal":
            reward = np.random.lognormal(mean = self.mu[action], sigma = 1) + self.rw_increment[action]

        regret = max(self.mu) - reward

        # updates increment by adding rv ~ Norm(0, 0.1)
        self.rw_increment += np.random.normal(0, 0.1, size = self.k) 

        return reward, regret

    def reset(self):
        self.rw_increment = np.zeros(size)
        return 0