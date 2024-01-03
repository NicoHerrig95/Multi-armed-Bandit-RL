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


    def __init__(self, env, q_initialisation = None, name = None):
        super().__init__(env, q_initialisation, name) # init from base_class_bandit

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

    def __init__(self, env, epsilon, q_initialisation = None, name = None):
        self.epsilon = epsilon
        super().__init__(env, q_initialisation, name) # init from base_class_bandit

    def select_action(self):

        rv = np.random.binomial(1, self.epsilon, 1)
    
        # choosing greedy action if rv = 0
        if rv == 0:
            a = super().greedy_action() # greedy action selection from base class
        # choosing non-greedy action of rv = 1
        elif rv == 1:
            a = super().random_action() # random action selection from base class

        return a       



