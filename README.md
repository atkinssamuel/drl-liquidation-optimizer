# drl-liquidation-optimizer
## Conda Environment Setup
**Create environment from environment.yml**:  
*From base directory:*  
```conda env create -f ./environment.yml```

**Update environment from environment.yml**:  
*From base directory and after activating existing environment:*  
```conda env update --file ./environment.yml```

## Current Goals
1. Finish DDPG implementation and benchmark it with current solution
2. Modify formulation to produce a new agent consistent with 5.2.2 of the HFT book (119 of .pdf)
