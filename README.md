# DRL Liquidation Optimizer
## Conda Environment Setup
**Create environment from environment.yml**:  
*From base directory:*  
```conda env create -f ./environment.yml```

**Update environment from environment.yml**:  
*From base directory and after activating existing environment:*  
```conda env update -f ./environment.yml```

## Project Execution
The entry point of the program for the DDPG algorithm is  ```run_ddpg.py``` and the entry point of the program for the PPO algorithm is ```run_ppo.py```. 

**CLI**:  
Activate the ```drl-liquidation-optimzer``` environment and ensure that it is up to date with the ```environment.yml``` file. Then, run either ```run_ddpg.py``` by executing ```python run_ddpg.py``` or ```run_ppo.py``` by executing ```python run_ppo.py``` from the base ```drl-liquidation-optimizer``` directory. 

**PyCharm**:  
To execute either of these programs in PyCharm, create a PyCharm configuration that executes either ```run_ddpg.py``` or ```run_ppo.py``` from the base ```drl-liquidation-optimizer``` folder using the ```drl-liquidation-optimizer``` conda environment defined in the ```environment.yml``` file. 



## Aiden Issue Closure Plan
- Assessment of convergence based on learning curve analysis (Ali)
- Behavioral analysis of the reward function in the face of non-stationarity (Ali)


- Building a MARL benchmark (Sam & William, EMRMGK-1769) - **April 20th**
	- Based on https://arxiv.org/abs/1312.7360
	- Use PPO
	- Explore hyper-parameter tuning:
		- PPO component
		- Architectural component
	- Think about how to create over-fitting scenarios
	- Assess stability of the reward curves
	- Assess dependence on the number of agents in MARL
	- Assess the effect of standardization
- Create reward correlation matrices for assessing environment over-fitting (Sam, EMRMGK-1770) - **May 4th**
	- Based on https://arxiv.org/abs/1711.00832


- Employ adversarial perturbations for robustness analysis (William) - **May 4th**
	- Based on https://arxiv.org/abs/2010.11388
	- Implemented on single agent (use the one Sam has already produced)
	

