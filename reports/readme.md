# README

## Overview
This project implements a reinforcement learning framework for solving a simple treasure hunt environment using various agents. The aim is for the hero to reach the treasure on a 10x10 grid-based world.
Agents include custom implementations of classical Q-learning and deep-learning agents from StableBaselines3.

## Installation

### Prerequisites
- Python 3.12 (might work with earlier versions, not tested)
- Docker (optional, for containerized execution)

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/CGuichaoua/rl-treasure.git
   cd rl-treasure
   ```
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

Alternatively, use Docker to build and run the project:
```bash
docker build -t treasure-hunt -f docker/Dockerfile .
docker run --rm -it treasure-hunt
```

## Running the Code

### Training an Agent
Run the following command to train an agent:
```bash
python main.py --environment base --agent DQN --epochs 1000 --timesteps 10000
```
- Replace `base` with the desired environment (e.g., `fixed` or `static`).
- Replace `DQN` with the desired agent (e.g., `tabular_q`, `near_sighted`).

### Rendering
To visualize the agent's actions, enable rendering:
```bash
python main.py --render
```

### Running a Pre-Trained Agent
Load and evaluate a pre-trained model:
```bash
python main.py --agent DQN --load-path ./models/dqn_treasure_model.zip
```

### Full --help output
```
usage: main.py [-h] [--environment {fixed,static,base}] [--agent {tabular_q,near_sighted,oblivious,DQN,PPO}] [--epochs EPOCHS]
               [--timesteps TIMESTEPS] [--render] [--force-train] [--load_model LOAD_MODEL]

Run agents on treasure hunt environments.

options:
  -h, --help            show this help message and exit
  --environment {fixed,static,base}
                        The environment to use. Can also be set via the TH_ENVIRONMENT env variable.
  --agent {tabular_q,near_sighted,oblivious,DQN,PPO}
                        The agent to run. Can also be set via the TH_AGENT env variable.
  --epochs EPOCHS       Number of epochs to train. Can also be set via TH_EPOCHS env variable.
  --timesteps TIMESTEPS
                        Number of timesteps to train each epoch. Can also be set via TH_TIMESTEPS env variable.
  --render              Render the environment. Use for debugging in non-Docker runs.
  --force-train         Force a train run. Useful when loading a model.
  --load_model LOAD_MODEL
                        Path to a pre-trained model to load.
```

## Code Structure

### Design Choices
- **Environment Modularity**: Environments follow `gymnasium`'s `Env` interface and can easily be swapped in and out. There are several variations of the TreasureHunt environment with different behaviors, and new monster movement patterns can easily be introduced with new MonsterStrategy subclasses.
- **StableBaselines3 compatibility**: Agents follow `stable-baselines3`'s interface and the baseline agents are useable (but might require an observation space adapter like `FlattenTreasureWrapper`) due to their own constraints.
- **Configurable Execution**: `main.py` allows runtime configuration via command-line arguments and environment variables.

### Directories
- **`environment/`**: Contains all custom environments and wrappers.
  - `BaseTreasureHuntEnv`: Base implementation for treasure hunt mechanics.
  - `FixedTreasureHuntEnv`: Simpler version with a set initial position.
  - **`monster_strategy`**: Implements monster strategies
    - `MonsterMovementStrategy`: Abstract base class for monster strategy
    - `StationaryStrategy`: Monsters are immobile traps
    - `RandomMovementStrategy`: Monsters wander around slowly and randomly
  - `FlattenTreasureWrapper`: Converts complex observations into a flattened space for compatibility with certain agents.
  - 
- **`agent/`**: Implements RL agents and respective environment reducers.
  - `TabularQLearner`: Basic tabular Q-Learning implementation.
  - `SimplifierQLearner`: Tabular Q-Learning, but agents reduce the observation space first.
  - **`env_reducer/`**: Environment reducers for the SimplifierQLearner agent
    - `EnvironmentReducer`: Abstract base class for the reducer interface
    - `ObliviousReducer`: Remove monsters from the observation
    - `NearSightedReducer`: Monster's relative position is encoded as near/fear in each direction
- **`utils/`**: Contains utility functions and classes.
  - `RLRunner`: Handles training, evaluation, and results saving.
  - `AdaptiveRLRunner`: Subclasss of RLRunner with dynamic evaluation length`
  - `run_with_render`: Helper function to watch an agent in an environment
- **`main.py`**: Entry point for running experiments.

