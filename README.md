# Multi-armed bandit environment and algorithms
> Python code library, containing reinforcement learning environment implementations of multiarmed bandit problems, as well as agents from the current literature to solve the problem. The repository further contains base classes for both the bandit environment (based on the `Env` class from OpenAI gym) and the agent, enabling users to easily build their own environment and agent.
> The scope of this project is to provide a comprehensive but easy-to-use environment for stochastic analysis of the problem.
> **Please note that this a working project**

## Table of Contents
* [Documentation](#documentation)
* [Setup](#setup)
* [Usage](#usage)
* [Acknowledgements](#acknowledgements)
<!-- * [License](#license) -->


## Documentation
The documentation shall give a basic understanding of bandit algorithms, environments and agents in general. The documentation can be accessed via the [Documentation file](Documentation_v1.pdf).


## Setup
The repository currently is only available via git. To clone the repository, please execute via Terminal/PowerShell:
```sh
git clone https://github.com/NicoHerrig95/Multi-armed-Bandit-RL.git
```
Then, create and activate a virtual environment.  
```sh
# virtual environment with name virt_env
python3 -m venv virt_env
# following line is for MacOS/Linux users
source virt_env/bin/activate
# ALTERNATIVE for windows
virt_env/Scripts/activate
```
Install dependencies via requirments.txt
```sh
pip install -r requirements.txt
```
-> Setup is now finished.


## Usage
A workflow (running an experiment) mainly consists of three components:
- **The Environment** (defining the problem)
- **The Agent** (defining the subject to solve the problem)
- **The 'Bandit'** (orchestrating the interactions between the agent and the environment)
Details for environments and agents can be found [here](Documentation_v1.pdf). The Bandit itself is an instance of the `Multiarmed_Bandit` object, defined [here](src/multiarmed_bandit.py). Generally, experiments can be executed as following:\


**1. Imports**
```python
# Imports depend on which agents and environments you want to use.

import sys
from src.bandit_environments import stochastic_bandit # environment import
from src.multiarmed_bandit import Multiarmed_Bandit
from src.agents import EpsilonGreedyAgent, RandomAgent, UCBAgent # agents import
```

**2. Initiating the environment,agent and bandit**
```python
# defining problem / environment
env = stochastic_bandit(size = 5) # 5-armed stochastic bandit

# defining agents
agent = EpsilonGreedyAgent(env = env, epsilon= 0.2, name = "Epsilon Agent")

# defining the bandit object. Takes the instances for agent and environment as input
bandit = Multiarmed_Bandit(environment=env, agent=agent)
```

**3. Run the experiemtn -> executing the bandit**
```python
# defining how much iterations the algorithm shall run
n_episodes = 1000 
# execution
bandit_1.execute(episodes=n_episodes) 
```

**4. Results -> running analytics**
The `Multiarmed_Bandit` class provides a number of callable functions for analysing the experiment's results, enabling comparison between several algorithms. Functions are listed below.\
- average_rewards() prints or plots the average rewards achieved by the agent over time.
- average_regrets() prints or plots the average regrets achieved by the agent over time.
- selected_actions() prints or plots the selected actions.
- total_results() prints the accumulated total results, meadured in total rewards and total regrets.
Each of these analytical functions have take the option `plot` as argument. `plot = False` prints the results over time, while `plot = True` plots a visualisation.
```python
# EXAMPLE
# This will plot the average achieved by the agent over time.
bandit.average_rewards(plot=True)
```

*An comprehensive example workflow can be found in [here](workflow_example.ipynb).*


## Acknowledgements
Credits to Richard S. Sutton and Andrew G. Barto (2018). Book: [Amazon-Link](https://www.amazon.de/-/en/Richard-S-Sutton/dp/0262039249/ref=sr_1_1?crid=1XHYMM96E466H&dib=eyJ2IjoiMSJ9.MVBhGDC2mDNtCrGRLs9ttAdjPmpi5wI8sY3XAIzI1Gmlioi8beByH4BrWHnz8C7G-tPZT50G23ly8B4UBFscgFzY2s17CUKnS4qVQY__rePu64Mh6cmqXHEf_lEY15FYgaUqjJYj5xTJXMOwtsjYMibl2il_V4xzwNq-O9BXHsaK2cVQeK-sTkfJpHIv4c9uStfs4_sZM7mIYEOttzG-cHLH3t8H7xqR2qo2OLJ8ewQ.IXuijPvrVFmPCMvoHK2nx5kOamz0W3xCr6Tk0dLkWxM&dib_tag=se&keywords=reinforcement+learning&qid=1709999088&sprefix=Reinfor%2Caps%2C122&sr=8-1)

