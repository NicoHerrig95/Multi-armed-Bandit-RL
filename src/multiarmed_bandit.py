##################################################################################################
##################################### -- BANDIT WRAPPER -- #######################################
##################################################################################################
import matplotlib.pyplot as plt
import numpy as np 

class Multiarmed_Bandit(object):
    '''
    Multiarmed Bandit object, can be used as a wrapper for training/executing an agent on a defined environment. 
    Requires a bandit environment and a bandit agent as input.
    
    Arguments:
    ----------
    environment: An initialised bandit environment 
    agent: An initialised bandit agent
    '''

    def __init__(
        self, 
        environment, 
        agent):

        self.agent = agent
        self.env = environment

        # getting variables from agent and environment
        self.env_mu = environment.mu
        self.k = environment.k

        

    def execute(self, episodes, out = True):
        '''
        CORE FUNCTIONALITY - executes the Multiarmed Bandit. 

        Arguments:
        ----------
        episodes: Number of episodes the agent is trained on the environment. Type -> [int]
        '''

        if not isinstance(out, bool):
            raise ValueError("out must be either True or False")
            
        # storers
        rewards = []
        avg_rewards = []
        regrets = []
        avg_regrets =[]
        avg_reward = 0
        avg_regret = 0
        n = 0 # counter

        for _ in range(episodes):

            n += 1 # episode count update

            a = self.agent.select_action()
            reward, regret = self.env.step(action = a)
            # update q-estimates
            self.agent.update_estimates(reward = reward, action=a)
            # print outcome from current episode
            if out == True:
                print(f"episode: {n} -- action: {a+1} -- reward: {reward} -- regret: {regret}")


            rewards.append(reward)
            regrets.append(regret)
            avg_reward = avg_reward + (1/n) * (reward - avg_reward)
            avg_rewards.append(avg_reward)
            avg_regret = avg_regret + (1/n) * (regret - avg_regret)
            avg_regrets.append(avg_regret)
        print("----- Execution succeeded -----")

        # results object for further usage
        self.results = {
            "rewards" : rewards,
            "regrets" : regrets,
            "avg rewards": avg_rewards,
            "avg regrets": avg_regrets
        }



    ##### VISUALISATION FUNCTIONS #####
    def average_rewards(self, plot = False):
        '''
        Returns either a list or a plot of average rewards over time.

        Arguments:
        ----------
        plot: Indicates whether or not average rewards are plotted. Type -> bool
        '''

        if not isinstance(plot, bool):
            raise ValueError("plot must be one of [True, False]")

        if plot == False: 
            return self.results['avg rewards']
        else: 
            # working on plot
            fig, ax = plt.subplots()
            ax.plot(self.results['avg rewards'])

            ax.set_ylabel('rewards')
            ax.set_xlabel("episode")
            ax.set_title('average rewards')
            plt.show()

    def average_regrets(self, plot = False):
        '''
        Returns either a list or a plot of average regrets over time.

        Arguments:
        ----------
        plot: Indicates whether or not average rewards are plotted. Type -> bool
        '''

        if not isinstance(plot, bool):
            raise ValueError("plot must be one of [True, False]")

        if plot == False: 
            return self.results['avg regrets']
        else:
            fig, ax = plt.subplots()
            ax.plot(self.results['avg regrets'])

            ax.set_ylabel('regrets')
            ax.set_xlabel("episode")
            ax.set_title('average regrets')
            plt.show()


    def selected_actions(self, plot = False):
        '''
        Returns either a list or a plot of selected action for current bandit execution.

        Arguments:
        ----------
        plot: Indicates whether or not average rewards are plotted. Type -> bool
        '''

        if not isinstance(plot, bool):
            raise ValueError("plot must be one of [True, False]")


        counts = self.agent.statistics()['action counts']

        if plot == False: 
            return counts

        if plot == True:

            fig, ax = plt.subplots()
            actions = [str(i+1) for i in (list(range(len(counts))))]
            ax.bar(actions, counts)

            ax.set_ylabel('total count')
            ax.set_xlabel("action")
            ax.set_title('action counts')
            plt.show()


    def total_results(self):
        ''' 
        Returns the total results.
        '''
        print(f'total rewards: {sum(self.results["rewards"])}')
        print(f'total regret: {sum(self.results["regrets"])}')




    def return_agent(self):
        ''' 
        Returns the agent of the Bandit object.
        '''

        return self.agent