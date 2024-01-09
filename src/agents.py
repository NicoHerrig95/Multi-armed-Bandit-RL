from utilities import *
import numpy as np 




class RandomAgent(base_class_bandit):
    '''
    Bandit Agent, selects actions on random

    Arguments
    ---------
    env: Bandit environment
    q_initialisation: initial q-estimates. Type -> int, float, complex, NoneType

    '''


    def __init__(self, env, q_initialisation = None, valuation_method = "average", alpha = None, name = None):
        self.t = 0
        super().__init__(env, q_initialisation, valuation_method, alpha, name) # init from base_class_bandit

    def select_action(self):
        a = super().random_action()
        return a  




class EpsilonGreedyAgent(base_class_bandit):
    '''
    Bandit Agent, uses epsilon-greedy method for action selection.

    Arguments
    ---------
    env: Bandit environment
    epsilon: epsilon value, where P(action = greedy) = 1 - epsilon. Type -> int, float, complex. Range -> [0,1]
    q_initialisation: initial q-estimates. Type -> int, float, complex, NoneType

    '''

    def __init__(self, env, epsilon, q_initialisation = None, valuation_method = "average", alpha = None, name = None):

        if not isinstance(epsilon, (int, float, complex)):
            raise ValueError("epsilon must be one of type [int, float, complex]")

        if epsilon < 0 or epsilon > 1:
            raise ValueError("epsilon must be in [0, 1]")

        self.epsilon = epsilon
        super().__init__(env, q_initialisation, valuation_method, alpha, name) # init from base_class_bandit

    def select_action(self):

        rv = np.random.binomial(1, self.epsilon, 1)
    
        # choosing greedy action if rv = 0
        if rv == 0:
            a = super().greedy_action() # greedy action selection from base class
        # choosing non-greedy action of rv = 1
        elif rv == 1:
            a = super().random_action() # random action selection from base class

        return a       




class UCBAgent(base_class_bandit):



    def __init__(self, env, c, name = None):

        if not isinstance(c, (int, float, complex)):
            raise ValueError("c must be one of type [int, float, complex]")

        if c < 0:
            raise ValueError("epsilon must a positive number")
        self.c = c

        self.t = 0 
        super().__init__(env, name)


    def select_action(self):
        
        # updating t
        self.t += 1
        UCB_value = 0
        UCB_table = []

        for i in range(self.k): 
            # UCB equation
            UCB_value = self.actions[i].q + self.c * (np.sqrt(np.log(self.t) / self.actions[i].n))
            UCB_table.append(UCB_value)

        a = np.argmax(UCB_table)

        return a 


        
        

        
        