"""Utility functions for the treasure hunt project."""

import pygame
import os
from datetime import datetime
import pickle

import numpy as np
import matplotlib.pyplot as plt


class RLRunner:
    """Class to handle running training and testing an agent."""

    def __init__(self, agent, env, *,
                 total_epochs=10000, eval_interval=1000, eval_episodes=5, verbose=True,
                 experiment_name=None, final_test_episodes=1000):
        self.agent = agent
        self.env = env
        self.total_epochs = total_epochs
        self.eval_interval = eval_interval
        self.eval_episodes = eval_episodes
        self.final_test_episodes = final_test_episodes
        self.verbose = verbose
        if experiment_name is None:
            self.experiment_name = f"{agent.__class__.__name__}_{
                env.__class__.__name__}"
        else:
            self.experiment_name = experiment_name

        self.reward_history = []
        self.last_rewards = []

    def train_agent(self):
        """Train the agent with regular evaluation loops."""
        total_epochs = self.total_epochs
        for epoch_no in range(total_epochs):
            self.train_agent_epoch()
            if self.verbose:
                print(f"Epoch {epoch_no +
                               1}/{total_epochs} - Training complete")
            self.test_agent()
            if self.verbose:
                print(f"Epoch {epoch_no +
                               1}/{total_epochs} - Mean reward: {self.reward_history[-1]}")

    def train_agent_epoch(self):
        """Run an epoch of training."""
        self.env.reset()
        self.agent.learn(total_timesteps=self.eval_interval)

    def test_agent(self, final_test=False):
        """Test the agent's performance."""
        eval_episodes = self.final_test_episodes if final_test else self.eval_episodes

        rewards = []
        for _ in range(eval_episodes):
            obs, _ = self.env.reset()
            episode_reward = 0
            done = False
            while not done:
                action, _ = self.agent.predict(obs, deterministic=True)
                obs, reward, done, truncated, _ = self.env.step(action)
                episode_reward += reward
                done = done or truncated
            rewards.append(episode_reward)
        self.last_rewards = rewards
        mean_reward = np.mean(rewards)

        self.reward_history.append(mean_reward)

    def save_results(self):
        """Save the reward history and agent."""
        # Get current timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Create results directory with timestamp subfolder if it doesn't exist
        results_dir = os.path.join('results', 'qlearner-static', timestamp)
        os.makedirs(results_dir, exist_ok=True)

        # Save reward history
        with open(os.path.join(results_dir, 'reward_history.pkl'), 'wb') as f:
            pickle.dump(self.reward_history, f)

        # Save mean reward
        with open(os.path.join(results_dir, 'mean_reward.txt'), 'w', encoding='utf8') as f:
            f.write(str(self.reward_history[-1]))

        # Save rewards
        with open(os.path.join(results_dir, 'rewards.pkl'), 'wb') as f:
            pickle.dump(self.last_rewards, f)

        # Save agent
        self.agent.save(os.path.join(results_dir, 'agent'))

        print(f"Results saved to the '{results_dir}' folder.")

    def plot_results(self):
        """Plot reward history"""
        plt.plot(self.reward_history)
        plt.xlabel("Epochs")
        plt.ylabel("Mean Reward")
        plt.title("Reward Progress Over Training")
        plt.show()


class AdaptiveRLRunner(RLRunner):
    """Class to handle running training and testing an agent with adaptive parameters."""

    def __init__(self, agent, env, *,
                 total_epochs=10000, eval_interval=1000, eval_episodes=5, verbose=True,
                 target_std_ratio=.5, adapt_window=10):
        super().__init__(agent, env, total_epochs=total_epochs, eval_interval=eval_interval,
                         eval_episodes=eval_episodes, verbose=verbose)
        self.target_std_ratio = target_std_ratio
        self.adapt_window = adapt_window

    def test_agent(self, final_test=False):
        """Train the agent with adaptive evaluation intervals."""
        super().test_agent()
        if not final_test:
            self.adapt_eval_interval()

    def adapt_eval_interval(self):
        """Adapt the evaluation episodes based on the standard deviation of the last rewards."""
        if len(self.reward_history) < self.adapt_window:
            return
        std_inner = np.std(self.last_rewards)
        std_inter = np.std(self.reward_history[-self.adapt_window:])
        std_ratio = std_inter / std_inner
        if std_ratio > self.target_std_ratio:
            self.eval_episodes += int(np.ceil(self.eval_episodes * 0.1))
        elif self.eval_episodes > 2 and std_ratio < self.target_std_ratio * 0.5:
            self.eval_episodes -= int(np.ceil(self.eval_episodes * 0.1))
        if self.verbose:
            print("Adapted evaluation episodes to "
                  f"{self.eval_episodes} based on std ratio {std_ratio}")


def run_with_render(env_human, agent, n_episodes=10):
    """Run the agent in the environment with rendering."""

    for episode_no in range(n_episodes):
        obs, info = env_human.reset()  # Reset the environment if the episode ends
        for _ in range(100):  # Add a limit
            action, _ = agent.predict(obs)  # Get the action from the agent
            obs, reward, terminated, truncated, info = env_human.step(action)
            env_human.render()  # Render the environment
            pygame.time.delay(100)  # Delay for 200 ms
            if terminated or truncated:
                break
