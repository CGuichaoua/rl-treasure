# Code structure

2 main submodules: agent and environment



## Experiments

### First iteration (sanity check)
- No randomness in environment 
- 2 monsters
- No monster movement
- Tabular Q-learner

Typically converges after about 10 epochs of 1000 timesteps.


### Second iteration :
- **Random initialisation of the environment**
- 2 monsters
- Still no monster movement
- Tabular Q-learner

** Want to check the convergence time**


### Third iteration :
- Random initialisation of the environment
- 2 monsters
- **Monsters move randomly**
- Tabular Q-learner

** Rewards are no longer deterministic. How much harder does it get to learn? **

