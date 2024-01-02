import numpy as np


class base_class_bandit(object):


    def __init__(self, env):

    # list of actions the agent can take.
    # is element of [0, k-1]
        self.actions = [] 
        self.k = env.k
        for _ in range(self.k): 
            self.actions.append(Action(initial_value = None))

        #self.value_method = value_method


    def test_init(self):
        print("Sample estimate",self.actions[1].q)
        print("length of action vector: ", len(self.actions))

    def select_action(self):
        pass


    def greedy(self):

        greedy_table = []
        for i in range(self.k):
            greedy_table.append(self.actions[i].q) 

        a = np.argmax(greedy_table)
        return a
        


    def q_estimation(self, reward, action):
        self.actions[action].n += 1 # update action counter

        # Method: average value estimation
        self.actions[action].q = self.actions[action].q + (1/self.actions[action].n) * (reward - self.actions[action].q)
