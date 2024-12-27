import argparse
import os
from gymnasium import make
from stable_baselines3 import DQN, PPO

from .environment import BaseTreasureHuntEnv, FixedTreasureHuntEnv
from .agent import SimplifierQLearner, TabularQLearner
from .agent.env_reducer import NearSightedReducer, ObliviousReducer
from .environment import FlattenTreasureWrapper
from .utils import AdaptiveRLRunner

ENVIRONMENTS = {
    "fixed": "FixedTreasureHunt-v0",  # Fixed environment for debugging
    "static": "StationaryMonsterTreasureHunt-v0",  # Stationary monsters environment
    "base": "RandomMonsterTreasureHunt-v0",  # Random monsters environment
}


def make_agent(agent_name, env):
    if agent_name == "tabular_q":
        return TabularQLearner(env)
    elif agent_name == "near_sighted":
        return SimplifierQLearner(env, NearSightedReducer(env.unwrapped))
    elif agent_name == "oblivious":
        return SimplifierQLearner(env, ObliviousReducer(env.unwrapped))
    elif agent_name == "DQN":
        return DQN("MlpPolicy", FlattenTreasureWrapper(env))
    elif agent_name == "PPO":
        return PPO("MlpPolicy", FlattenTreasureWrapper(env))
    else:
        raise ValueError(f"Unknown agent: {agent_name}")


VALID_AGENTS = ["tabular_q", "near_sighted", "oblivious", "DQN", "PPO"]


def main():
    parser = argparse.ArgumentParser(
        description="Run agents on treasure hunt environments.")
    parser.add_argument("--environment", default=os.getenv("ENVIRONMENT", "base"), choices=ENVIRONMENTS.keys(),
                        help="The environment to use. Can also be set via the ENVIRONMENT env variable.")
    parser.add_argument("--agent", default=os.getenv("AGENT", "near_sighted"), choices=VALID_AGENTS,
                        help="The agent to run. Can also be set via the AGENT env variable.")
    parser.add_argument("--epochs", type=int, default=int(os.getenv("EPOCHS", 1000)),
                        help="Number of epochs to train/evaluate. Can also be set via EPOCHS env variable.")
    parser.add_argument("--timesteps", type=int, default=int(os.getenv("TIMESTEPS", 1000)),
                        help="Number of timesteps to train/evaluate. Can also be set via TIMESTEPS env variable.")
    parser.add_argument("--render", action="store_true",
                        help="Render the environment. Use for debugging in non-Docker runs.")
    args = parser.parse_args()

    env_id = ENVIRONMENTS[args.environment]
    env = make(env_id, render_mode="human" if args.render else None)

    agent = make_agent(args.agent, env)
    runner = AdaptiveRLRunner(agent, env,
                              total_epochs=args.timesteps)
    runner.train_agent()
    runner.test_agent(final_test=True)
    runner.save_results()

    env.close()


if __name__ == "__main__":
    main()
