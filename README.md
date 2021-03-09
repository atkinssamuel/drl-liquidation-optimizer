# drl-liquidation-optimizer
## Conda Environment Setup
**Create environment from environment.yml**:  
*From base directory:*  
```conda env create -f ./environment.yml```

**Update environment from environment.yml**:  
*From base directory and after activating existing environment:*  
```conda env update --file ./environment.yml```

## Current Goals
1. Get Jaimungal environment working with chosen parameters
2. Calculate objective function for Jaimungal environment
3. Simulate M trials for Jaimungal environment and benchmark TWAP with optimal strategy
