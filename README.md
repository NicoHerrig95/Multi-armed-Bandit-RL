# Multi-armed bandit environment and algorithms
> Python code library, containing reinforcement learning environment implementations of multiarmed bandit problems, as well as agents from the current literature to solve the problem. The repository further contains base classes for both the bandit environment (based on the `Env` class from OpenAI gym) and the agent, enabling users to easily build their own environment and agent.
> The scope of this project is to provide a comprehensive but easy-to-use environment for stochastic analysis of the problem.
> **Please note that this a working project**

## Table of Contents
* [Content of Repository](#content-of-repository)
* [Setup](#setup)
* [Usage](#usage)
* [Project Status](#project-status)
* [Acknowledgements](#acknowledgements)
<!-- * [License](#license) -->


## Content of Repository
### Base Notation
$k$: number of bandit arms, or size of the bandit environment\
$t$: discrete time step\
$a$: denotes an action. $a \in k$\
$A_t$: denotes the action taken at time point $t$\
$R_{t,a}$: realized (ex-post) reward at time point $t$, given that $A_t = a$\
$Q_t(a)$: reward estimate for action $a$ at time point $t$\
$q_*(a) = \mathbb{E}(R_t|A_t = a)$: expected reward of action $a$ at time point $t$\
$N_t(a)$: number of times action $a$ has been selected until time point $t$


### Environments
At the moment, the repository contains 3 bandit environments.

1. **Stochastic Bandit Environment `stochastic_bandit`**\
The environment `stochastic_bandit` is a stationary bandit problem, where the reward of each arm is a random variable coming from either a gaussian normal or a log-normal distribution.\
When initialising the environment, the expected rewards for each action $q_(a)*$ from a normal distribution, where $\mu$ and $\sigma$ are argument inputs of the environment class (`stochastic_moments = [0,1]` per default.)\
\
**Arguments**\
`size`: the number of bandit arms $k$\
`reward_distribution`: defines the theoretical distribution of rewards. Options: `["gaussian", "lognormal"]`.\
`stochastic_moments`: Defines the first and second moment of the normal distribution the expected rewards $q_*(1),...,q_*(k)$ are sampled from ($\mu=0$ and $\sigma=1$ by default).\
\
*Example of a 5-armed bandit:* If the environment is initialised with `stochastic_moments = [1,2]`, the expected returns for each of the 5 reward distributions $q_*(1),...,q_*(5)$ are random variables from a normal distribution with $\mu = 1$ and $\sigma = 2$ ($N(1,2)$). After the initialisation, let $q_*(5) = 0.2$. Subsequently, $R_{t,5} \sim \mathcal{N}(0.2,1)$ in case of normally distributed rewards ($\sim \mathcal{logN}(0.2,1)$ applies if the user chose the log-normal distribution).  




## Setup
**The library will be available via pip once finalised.**


## Usage
1. Initialising the environment\

`write-your`\
`-code-here`


## Project Status
Project is: *in progress* 




## Acknowledgements
Give credit here.
- This project was inspired by...
- This project was based on [this tutorial](https://www.example.com).
- Many thanks to...

