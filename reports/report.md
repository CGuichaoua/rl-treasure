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

The learner now takes over 100000 epochs to achieve a perfect result. It takes a long time to even learn that it shouldn't run into walls, because most situations it encounters are brand new.

### Third iteration :
- Random initialisation of the environment
- 2 monsters
- Still no monster movement
- **Oblivious Q-learner** : ignores part of the observation
- **Near-sighted Q-learner** : sees only if a monster is close in any given direction and its position

The Oblivious Q-learner learns as fast as the tabular with fixed monsters, but it caps out in terms of performance because it sometimes runs into monsters.

The Near-sighted Q-learner is slower to learn, and at around 2500 epochs, it's starting to stabilize around the performance of the Oblivious Q-learner. More time learning might still lead it to win more often

### Fourth iteration :
- Random initialisation of the environment
- 2 monsters
- **Monsters move randomly**
- Near-sighted Q-learner, Oblivious Q-Learner

Moving monsters make it harder for the oblivious agent to rush to the end as monsters cover more ground (simply from having a turn of their own). The near-sighted agent has similar results to the static case

