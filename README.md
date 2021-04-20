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

#### Resources
- https://www.youtube.com/watch?v=kWHSH2HgbNQ
- https://www.youtube.com/watch?v=2Epn__SRHns