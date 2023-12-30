import numpy as np 
import matplotlib.pyplot as plt



class multiarmed_bandit:
    '''
    A flexible class, producing an object which acts as a bandit problem.
    --------------------------------------------------------------------- 
    Arguments:
    k: number of arms/actions
    action_selection: method for selecting respective action to be executed ["epsilon_greedy", "UCB"]
    epsilon: probability of choosing non-greedy action
    iterations: number of iterations
    mu: mean values for actual rewards, which follow a defined theoretical probability distribution ["random", list of length k]
    action_value_method: the specified method for calculating the estimated rewards ["average", "weighted"]
    alpha: step size parameter for action value method "weighted"
    stationarity: indicates whether bandit problem shall behave stationary or non-stationary ["True","False"]
    initialisation: initial values for Q estimates ["default", "optimistic"]
    init_values: if initialisation == "optimistic", the model requires a list (length k) of initial values for Q estimates
    seed: integer value for random seed generation
    '''
    

    def __init__(
        self,
        k = 10,
        action_selection = "epsilon_greedy",
        epsilon = 0,
        UCB_constant = 2,
        mu = "random", 
        iterations = 1000, 
        action_value_method = "average", 
        alpha = None, 
        stationarity = True, 
        initialisation = "default",
        init_values = None,
        seed = None):

        # random seed
        np.random.seed(seed)

        # K
        if not isinstance(k, int) or k < 0: raise ValueError("k must be an integer and > 0")
        self.k = k

        # Action Selection Method
        action_selection_options = ["epsilon_greedy", "UCB"]
        if action_selection not in action_selection_options:
            raise ValueError("value for action_selection must be one of: {}".format(action_selection_options))
        self.action_selection = action_selection

        # table for storing upper confidenceb bounds
        if action_selection == "UCB":
            self.UCB_table = np.zeros(k)
        
        # Epsilon
        if not isinstance(epsilon, (int, float, complex)) or isinstance(epsilon, bool):
            raise ValueError("epsilon must be numeric")
        if epsilon > 1 or epsilon < 0: raise ValueError("epsilon must be in [0,1]")
        self.epsilon = epsilon

        # UCB Constanst
        if not isinstance(UCB_constant, (int, float, complex)) or isinstance(UCB_constant, bool):
            raise ValueError("UCB constant must be numeric")
        if UCB_constant <= 0: raise ValueError("UCB constant must be greater than 0")
        self.c = UCB_constant

        # Alpha
        if alpha is not None:
            if not isinstance(alpha, (int, float, complex, NoneType)) or isinstance(alpha, bool):
                raise ValueError("alpha must be numeric")
            if alpha > 1 or alpha < 0: raise ValueError("alpha must be in [0,1]")
        self.alpha = alpha

        # Iterations
        # testing for object type
        if not isinstance(iterations, int) or iterations <= 0:
            raise ValueError("iterations must be an integer and > 0")
        self.iters = iterations


        ##### REWARD ESTIMATES Q(a) #####
        # initial values
        initalisation_options = ["default", "optimistic"] #list of possible options for initialisation
        # testing if initialisation is element of options list
        if initialisation not in initalisation_options:
            raise ValueError("value for initialisation must be one of: {}".format(initalisation_options))

        # Init values
        # check if init_values is list of numeric if initialisation method is not "default"
        if initialisation != "default":
            if not all(isinstance(x, (int, float, complex)) for x in init_values):
                raise ValueError("init_values must be a list of numerics")
            # check if init_values is of length k
            if len(init_values) != k:
                raise ValueError("init_values must be of length k")  
        
        # assigning initial values
        if initialisation == "default":
            self.q = np.zeros(k)
        elif initialisation == "optimistic" and isinstance(init_values, list):
            self.q =np.asfarray(init_values)
        
        # Action Value Method
        value_method_options = ["average", "weighted"]
        # checking if method is element of possible options
        if action_value_method not in value_method_options:
             raise ValueError("value for action_value_method mus be one of: {}".format(value_method_options))
        self.value_method = action_value_method # assigning method


        # Expected values (mu) for rewards
        # if mu shall be gaussian rv, sampling mu from N(0,1)
        if mu == "random":
            self.mu = np.random.normal(0,1,k)
        
        #if mu is inputted as list, converting to np.array
        if isinstance(mu, list):
            # object type check
            if not all(isinstance(x, (int, float, complex)) for x in mu):
                 raise ValueError("mu values must be a list of numerics") 
            self.mu = np.asfarray(mu)


        # stationarity boolean + storer
        if not isinstance(stationarity, bool):
            raise ValueError("stationarity must be either True or False")

        if stationarity == False:
            self.stationarity = False
            self.rw_increment = np.zeros(k)
        else:
            self.stationarity = True


        # iteration counter 
        self.n = 0 # initial step count
        # initial count for executing each of k actions/pulling arms
        self.action_n = np.full(k, 10e-8) # small float for avoiding division by 0 in UCB calculation
        # tracker
        self.q_tracker = np.zeros(shape=(self.iters, self.k)) # table for estimare tracking
        self.mu_tracker = np.zeros(shape=(self.iters, self.k))



        # TRUE/REALIZED REWARDS
        # array for storing realized reward values
        self.realized_reward = None # empty variable for realized reward
        self.mean_reward = 0 # variable for mean reward over time 
        self.rewards = np.zeros(iterations) # storer for realized reward per iteration
        self.rewards_avg = np.zeros(iterations) # storer mean reward over time 


        # REGRET
        self.realized_regret = None
        self.mean_regret = 0
        self.regrets = np.zeros(iterations)
        self.regrets_avg = np.zeros(iterations)

    

    def trigger(self):

        # ACTION SELECTION
        # Epsilon Greedy
        if self.action_selection == "epsilon_greedy":

            # generating a rv from binomial distribution with
            # P(rv = 1) = epsilon
            # rv=1 -> non-greedy execution
            # rv=0 -> greedy execution
            rv = np.random.binomial(1, self.epsilon, 1)
            # n = 0
            if self.n == 0:
                # randomly selects integer from array[0,...,k-1] with unit probability
                a = np.random.choice(self.k)
            
            # n > 1 
            # choosing greedy action if rv = 0
            elif rv == 0:
                a = np.argmax(self.q)
            # choosing non-greedy action of rv = 1
            elif rv == 1:
                a = np.random.choice(self.k)





        # Upper Confidence Bound
        if self.action_selection == "UCB":
            if self.n == 0:
                # randomly selects integer from array[0,...,k-1] with unit probability
                a = np.random.choice(self.k)

            elif self.n >= 1:
                self.UCB_table = self.q + self.c * np.sqrt(np.log(self.n) / self.action_n)
                a = np.argmax(self.UCB_table)






        # REALIZED REWARD
        if self.stationarity == True:
            # sampling actual reward from respective gaussian -> N(mu_a, 1)
            self.realized_reward = np.random.normal(self.mu[a], 1)
            # calculating random regret 
            self.realized_regret = max(self.mu) - self.realized_reward

        if self.stationarity == False:
            # adding random term to increment vector for random walk simulation
            self.rw_increment += np.random.normal(0,0.1, size = self.k) 
            self.realized_reward = np.random.normal(self.mu[a], 1) + self.rw_increment[a]
            # calculating random regret 
            self.realized_regret = max(self.mu) - self.realized_reward

        # UPDATING ACTION COUNTS
        self.n += 1 
        self.action_n[a] += 1

        # calculating and updating mean reward
        self.mean_reward = self.mean_reward + (1/self.n) * (self.realized_reward - self.mean_reward)

        # calculating mean regret
        self.mean_regret = self.mean_regret + (1/self.n) * (self.realized_regret - self.mean_regret)
       

        # VALUE UPDATE / Q ESTIMATES
        # sample-average method (for stationary case)
        if self.value_method == "average":
            self.q[a] = self.q[a] + (1/self.action_n[a]) * (self.realized_reward - self.q[a])
        # weighted method
        if self.value_method == "weighted":
            self.q[a] = self.q[a] + self.alpha * (self.realized_reward - self.q[a])
            

    def execute(self):
        # executing bandit for given iterations

        # loop over iterations/steps
        for i in range(self.iters):
            self.trigger()

            # storing results
            self.rewards[i] = self.realized_reward # storing the total realized reward in output vector
            self.rewards_avg[i] = self.mean_reward # storing the mean reward at t into corresponding storing vector
            self.regrets[i] = self.realized_regret
            self.regrets_avg[i] = self.mean_regret
            self.q_tracker[i] = self.q

            out = {
                "iteration ": self.n,
                "average reward": self.mean_reward,
                "average regret": self.mean_regret
                }

            print(out)

        
    def results(self):
        out = {
            "realized rewards":self.rewards,
            "average rewards":self.rewards_avg,
            "realized regrets":self.regrets,
            "average regrets":self.regrets_avg,
            "q-estimates":self.q_tracker,
            "mu-values":self.mu,
            "action counter": self.action_n.astype(int)
        }

        return(out)

    
    # list of all statistics
    def stats(self):

        stats = {
            "realized rewards":self.rewards,
            "average rewards":self.rewards_avg,
            "q-estimates":self.q,
            "mu-values":self.mu,
            "action counter":self.action_n,
            "epsilon": self.epsilon,
            "mu values":self.mu,
            "action_value_method":self.value_method,
            "number of arms":self.k,
            "alpha":self.alpha,
            "epsilon": self.epsilon,
            "iterations": self.iters,
            "step count": self.n,
            "action count":self.action_n.astype(int)
        }

        return stats


    # envorinment reset
    def reset(self):
        self.n = 0
        self.action_n = np.full(self.k, 10e-8)
        self.mean_reward = 0 
        self.realized_reward = 0 
        self.rewards_avg = np.zeros(self.iters)
        self.rewards = np.zeros(self.iters) # array for storing realized reward values
        self.q = np.zeros(self.k)
        self.q_tracker = np.zeros((self.iters, self.k))

        if self.n == 0 and self.mean_reward == 0:
            print("--- environment reset successfull ---")
