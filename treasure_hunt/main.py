"""Main script to run agents on treasure hunt environments."""
import argparse
import os
from gymnasium import make
from stable_baselines3 import DQN, PPO

from .environment import BaseTreasureHuntEnv, FixedTreasureHuntEnv
from .agent import SimplifierQLearner, TabularQLearner
from .agent.env_reducer import NearSightedReducer, ObliviousReducer
from .environment import FlattenTreasureWrapper
from .utils import AdaptiveRLRunner, run_with_render

ENVIRONMENTS = {
    "fixed": "FixedTreasureHunt-v0",  # Fixed environment for debugging
    "static": "StationaryMonsterTreasureHunt-v0",  # Stationary monsters environment
    "base": "RandomMonsterTreasureHunt-v0",  # Random monsters environment
}


def make_agent(agent_name, env, load_model=None):
    """Create an agent based on the agent name.
    Optionally load a pre-trained model.
    Return the agent and appropriately wrapped environment."""
    if agent_name == "tabular_q":
        agent = TabularQLearner(env)
    elif agent_name == "near_sighted":
        agent = SimplifierQLearner(env, NearSightedReducer(env.unwrapped))
    elif agent_name == "oblivious":
        agent = SimplifierQLearner(env, ObliviousReducer(env.unwrapped))
    elif agent_name == "DQN":
        env = FlattenTreasureWrapper(env)
        agent = DQN("MlpPolicy", env)
    elif agent_name == "DQN-smaller":
        env = FlattenTreasureWrapper(env)
        agent = DQN("MlpPolicy", env, policy_kwargs={"net_arch": [64, 64]})
    elif agent_name == "DQN-larger":
        env = FlattenTreasureWrapper(env)
        agent = DQN("MlpPolicy", env, policy_kwargs={"net_arch": [256, 256]})
    elif agent_name == "PPO":
        env = FlattenTreasureWrapper(env)
        agent = PPO("MlpPolicy", env)
    elif agent_name == "PPO-smaller":
        env = FlattenTreasureWrapper(env)
        agent = PPO("MlpPolicy", env, policy_kwargs={"net_arch": [64, 64]})
    elif agent_name == "PPO-larger":
        env = FlattenTreasureWrapper(env)
        agent = PPO("MlpPolicy", env, policy_kwargs={"net_arch": [256, 256]})
    else:
        raise ValueError(f"Unknown agent: {agent_name}")

    # Load pre-trained model if a path is provided
    if load_model:
        agent.load(load_model)

    return agent, env


VALID_AGENTS = ["tabular_q", "near_sighted",
                "oblivious", "DQN", "DQN-smaller", "DQN-larger", "PPO", "PPO-smaller", "PPO-larger"]


def main():
    parser = argparse.ArgumentParser(
        description="Run agents on treasure hunt environments.")
    parser.add_argument("--environment", default=os.getenv("TH_ENVIRONMENT", "base"), choices=ENVIRONMENTS.keys(),
                        help="The environment to use. Can also be set via the TH_ENVIRONMENT env variable.")
    parser.add_argument("--agent", default=os.getenv("TH_AGENT", "near_sighted"), choices=VALID_AGENTS,
                        help="The agent to run. Can also be set via the TH_AGENT env variable.")
    parser.add_argument("--epochs", type=int, default=int(os.getenv("TH_EPOCHS", 1000)),
                        help="Number of epochs to train. Can also be set via TH_EPOCHS env variable.")
    parser.add_argument("--timesteps", type=int, default=int(os.getenv("TH_TIMESTEPS", 10000)),
                        help="Number of timesteps to train each epoch. Can also be set via TH_TIMESTEPS env variable.")
    parser.add_argument("--render", action="store_true",
                        help="Render the environment. Use for debugging in non-Docker runs.")
    parser.add_argument("--force-train", action="store_true",
                        help="Force a train run. Useful when loading a model.")
    parser.add_argument("--load_model", type=str, default=None,
                        help="Path to a pre-trained model to load.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility.")
    parser.add_argument("--no-show", action="store_true",
                        help="Do not show the plot (useful for batch run)")

    args = parser.parse_args()

    env_id = ENVIRONMENTS[args.environment]
    env = make(env_id, render_mode="human" if args.render else None,
               max_episode_steps=500)

    agent, env = make_agent(args.agent, env, load_model=args.load_model)

    runner = AdaptiveRLRunner(agent, env,
                              total_epochs=args.epochs,
                              eval_interval=args.timesteps,
                              experiment_name=f"{args.agent}_{
                                  args.environment}",
                              seed=args.seed)
    if args.load_model and not args.force_train:
        print("Loaded pre-trained model")
        if args.render:
            print("Preloaded model and rendering is on: demo run")
            run_with_render(env, agent, n_episodes=10)
        else:
            print("Preloaded model and rendering is off: testing")
            # Only test when loading a model
            runner.test_agent(final_test=True)
            print(f"Final test results: {runner.reward_history[-1]}")

    else:
        if args.render:
            print("Rendering is on: demo run (before training)")
            run_with_render(env, agent, n_episodes=10)
        runner.train_agent()
        runner.test_agent(final_test=True)
        if not args.no_show:
            runner.plot_results(save=True)
        runner.save_results()

    env.close()


if __name__ == "__main__":
    main()
