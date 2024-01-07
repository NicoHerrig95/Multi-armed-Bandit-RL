##### UTILITIES #####
import numpy as np
import gym
from gym import spaces
import matplotlib.pyplot as plt



class Action(object):
    """
    Class object for an action the agent can take.
    
    Arguments:
    ----------
    initial_value: Initial values for q-estimates, either None or a numeric value. Type -> int, float, complex, NoneType 
    """

    def __init__(self, initial_value = None):


        if initial_value != None and not isinstance(initial_value, (int, float, complex)):
                raise ValueError("initial_value must be of types: [list, int, float, complex, NoneType]")

        # assigning q-estimates
        if initial_value == None:
            self.q = 0

        if initial_value != None:
            self.q = initial_value

        # counter for time action has been selected
        self.n = 0


##################################################################################################
##################################### -- BASE CLASSES -- #########################################
##################################################################################################

class base_class_bandit(object):
    '''
    Base class object for bandit agents. 

    Arguments:
    ----------
    env: Bandit environment
    q_initialisation: initial q-estimates. Type -> int, float, complex, NoneType
    valuation_method: Value-estimation method used for calculating Q-estimates. Must be one of ["average", "weighted"]
    alpha: Requried when method == "weighted". Scales the weighting for the last reward. 
           Must be within [0,1]. Type -> [int, float, complex]
    '''

    def __init__(self, env, q_initialisation = None, valuation_method = "average", alpha = None, name = None):

    # list of actions the agent can take.
    # is element of [0, k-1]
        
        self.k = env.k

        # ALPHA
        if not isinstance(alpha, (float, int, complex)) or alpha < 0 or alpha > 1:
            raise ValueError("alpha must be of type [int, float, complex] and in [0, 1]")
        self.alpha = alpha 

        # NAME
        if name != None:
            if not isinstance(name, str):
                raise ValueError("name must be of type [string]")
        self.name = name

        # VALUATION METHOD
        valuation_method_options = ["average", "weighted"]
        # checking if method is element of possible options
        if valuation_method not in valuation_method_options:
             raise ValueError("value for method must be one of: {}".format(valuation_method_options))
        self.method = valuation_method

        # check for method == "weighted"
        if valuation_method == "weighted":
            if not isinstance(alpha, (int, float, complex)):
                raise ValueError("for weighted method, alpha must be of Type [int, float, complex]")
            if alpha > 1 or alpha < 0:
                raise ValueError("alpha must be [0, 1]") 

        # Q INITIALISATION
        if q_initialisation != None and not isinstance(initial_value, (int, float, complex)):
            raise ValueError("q_initialisation must be of types: [list, int, float, complex, NoneType]")
        self.initial_value = q_initialisation

        # ACTIONS 
        # generating list of action objects
        self.actions = [] 
        for _ in range(self.k): 
            self.actions.append(Action(initial_value = self.initial_value))



    def select_action(self):
        '''
        Function for selecting an action. Method is dependent on implemented bandit algorithm.
        '''
        pass


    def greedy_action(self): 
        '''
        Selects action with highest q-estimate (greedy action).
        '''

        greedy_table = []
        for i in range(self.k):
            greedy_table.append(self.actions[i].q) 

        a = np.argmax(greedy_table)
        return a


    def random_action(self):
        '''
        selects random action.
        '''

        # selects an action randomly
        a = np.random.choice(self.k)

        return a


    def update_estimates(self, reward, action):
        '''
        Updates the Q-estimates based on the received reward and the action taken.

        Arguments:
        ----------
        reward: Reward received from the environment.
        action: Action taken for receiving the reward.
        '''

        # update action counter
        self.actions[action].n += 1 

        if self.method == "average":
            # Method: average value estimation
            self.actions[action].q = self.actions[action].q + (1/self.actions[action].n) * (reward - self.actions[action].q)

        if self.method == "weighted":
            self.actions[action].q = self.actions[action].q + self.alpha * (reward - self.actions[action].q)


    def statistics(self):
        action_counts = []
        for i in range(self.k): action_counts.append(self.actions[i].n)

        estimates = []
        for i in range(self.k): estimates.append(self.actions[i].q)

        out = {
            "action counts" : action_counts,
            "Q-estimates" : estimates
        }

        return out


    def reset(self):
        ''' 
        resets the agent
        '''
        self.actions = [] 
        for _ in range(self.k): 
            self.actions.append(Action(initial_value = self.initial_value))






class base_bandit_env(gym.Env):

    def __init__(self, size):
        
        # defining size of environment
        if not isinstance(size, int): 
            raise ValueError("size must be of type [int]")
        self.k = size


        # environment spaces
        self.action_space = spaces.Discrete(self.k)
        self.observation_space = spaces.Discrete(1)

    

    def step(self, action):
        # To be overwritten by specific bandit implementation
        pass

    def reset(self):
        # To be overwritten by specific bandit implementation
        return 0

    def render(self, mode='human', close=False):
        pass


