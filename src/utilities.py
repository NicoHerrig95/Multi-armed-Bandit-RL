##### UTILITIES #####
import numpy as np
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





class base_class_bandit(object):
    '''
    Base class object for bandit agents. 

    Arguments:
    ----------
    env: Bandit environment
    q_initialisation: initial q-estimates. Type -> int, float, complex, NoneType
    '''

    def __init__(self, env, q_initialisation = None, name = None):

    # list of actions the agent can take.
    # is element of [0, k-1]
        self.actions = [] 
        self.k = env.k

        if name != None:
            if not isinstance(name, str):
                raise ValueError("name must be of type [string]")
        self.name = name


        if q_initialisation != None and not isinstance(initial_value, (int, float, complex)):
            raise ValueError("q_initialisation must be of types: [list, int, float, complex, NoneType]")
        self.initial_value = q_initialisation

        for _ in range(self.k): 
            self.actions.append(Action(initial_value = self.initial_value))



    def select_action(self):
        '''
        Function for selecting an action. Method is dependent on implemented bandit algorithm.
        '''
    
        pass


    def greedy_action(self): 
        '''
        selects action with highest q-estimate (greedy action)
        '''

        greedy_table = []
        for i in range(self.k):
            greedy_table.append(self.actions[i].q) 

        a = np.argmax(greedy_table)
        return a


    def random_action(self):
        '''
        selects random action
        '''

        # selects an action randomly
        a = np.random.choice(self.k)

        return a


    def update_estimates(self, reward, action, method = "average", alpha = None):
        '''
        Update function for q-estimates
        '''

        method_options = ["average", "weighted"]
        # checking if method is element of possible options
        if method not in method_options:
             raise ValueError("value for method must be one of: {}".format(method_options))

        # check for method == "weighted"
        if method == "weighted":
            if not isinstance(alpha, (int, float, complex)):
                raise ValueError("for weighted method, alpha must be of Type [int, float, complex]")
            if alpha > 1 or alpha < 0:
                raise ValueError("alpha must be [0, 1]") 


        # update action counter
        self.actions[action].n += 1 

        if method == "average":
            # Method: average value estimation
            self.actions[action].q = self.actions[action].q + (1/self.actions[action].n) * (reward - self.actions[action].q)

        if method == "weighted":
            self.actions[action].q = self.actions[action].q + alpha * (reward - self.actions[action].q)




class Multiarmed_Bandit(object):

    def __init__(
        self, 
        environment, 
        agent):

        self.agent = agent
        self.env = environment

        # getting variables from agent and environment
        self.env_mu = environment.mu
        self.k = environment.K
        

    def execute(self, episodes, value_estimation = "average"):

        value_estimation_options = ["average", "weighted"]
        if value_estimation not in value_estimation_options:
             raise ValueError("value_estimation must be one of: {}".format(value_estimation_options))
        self.value_estimation = value_estimation

        # storers
        rewards = []
        avg_rewards = []
        regrets = []
        avg_regrets =[]
        avg_reward = 0
        avg_regret = 0
        n = 0 # counter

        for _ in range(episodes):

            n += 1

            a = self.agent.select_action()
            reward, regret = self.env.step(action = a)
            print("action:", a, "reward: ", reward, "regret:", regret)
            # update q-estimates
            self.agent.update_estimates(reward = reward, action=a, method = self.value_estimation)


            rewards.append(reward)
            regrets.append(regret)
            avg_reward = avg_reward + (1/n) * (reward - avg_reward)
            avg_rewards.append(avg_reward)
            avg_regret = avg_regret + (1/n) * (regret - avg_regret)
            avg_regrets.append(avg_regret)


        # results object for further usage
        self.results = {
            "rewards" : rewards,
            "regrets" : regrets,
            "avg rewards": avg_rewards,
            "avg regrets": avg_regrets
        }



    ##### VISUALISATION FUNCTIONS #####
    def average_rewards(self, plot = False):

        if plot == False: return self.results['avg rewards']

        if plot == True:
            plt.plot(self.results['avg rewards'])
            plt.show()


    def selected_actions(self, plot = False):

        out = []
        for i in range(self.k): out.append(self.agent.actions[i].n)

        if plot == False: return out     