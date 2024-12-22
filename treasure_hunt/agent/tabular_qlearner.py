"""Simple Q-learner using a table"""

from collections import defaultdict

import numpy as np
import gymnasium as gym


class TabularQLearner:
    """
    A simple Tabular Q-learning agent compatible with the Stable-Baselines3 interface.
    """

    def __init__(self, env: gym.Env, learning_rate=0.1, discount_factor=0.99,
                 exploration_rate=1.0, exploration_decay=0.995, min_exploration_rate=0.01):
        self.env = env

        # Learning parameters
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.min_exploration_rate = min_exploration_rate

        self.q_table = defaultdict(lambda: np.zeros(env.action_space.n))

    def _serialize_state(self, state: dict) -> tuple:
        """Convert the observation dict into a hashable state tuple."""
        return tuple(state.keys())

    def _select_action(self, state: tuple, deterministic: bool) -> int:
        """Select an action based on exploration or exploitation."""
        if deterministic or np.random.rand() > self.exploration_rate:
            return np.argmax(self.q_table[state])  # Exploit
        return self.env.action_space.sample()  # Explore

    def _update_q_value(self, state: tuple, action: int, reward: float, next_state: tuple):
        """Update the Q-value using the Q-learning formula."""
        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + self.discount_factor * \
            self.q_table[next_state][best_next_action]
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.learning_rate * td_error

    def _decay_learning_rate(self):
        """Decay exploration rate."""
        self.exploration_rate = max(self.min_exploration_rate,
                                    self.exploration_rate * self.exploration_decay)

    def train(self, total_timesteps=10000):
        """
        Train the agent using Q-learning.
        """
        state, _ = self.env.reset()  # Get the initial observation
        # Focus on the hero's position for tabular Q-learning
        state = state['hero_position']

        for _ in range(total_timesteps):
            # Epsilon-greedy action selection
            action = self._select_action(state, deterministic=False)
            next_state, reward, done, truncated, _ = self.env.step(action)
            next_state = self._serialize_state(next_state)

            self._update_q_value(state, action, reward, next_state)

            # Reset environment if done
            if done or truncated:
                state, _ = self.env.reset()
                state = self._serialize_state(state)
            else:
                state = next_state

            self._decay_learning_rate()

    def predict(self, observation, deterministic=True):
        """
        Predict an action given the observation.
        SB3-compatible interface.
        """
        state = self._serialize_state(observation)
        action = self._select_action(state, deterministic)
        return action, None

    def save(self, path):
        """
        Save the Q-table.
        """
        with open(path, 'wb') as f:
            # Convert defaultdict to dict for saving
            np.save(f, dict(self.q_table))

    def load(self, path):
        """
        Load the Q-table.
        """
        with open(path, 'rb') as f:
            q_table = np.load(f, allow_pickle=True).item()
            self.q_table = defaultdict(lambda: np.zeros(
                self.env.action_space.n), q_table)

    def learn(self, total_timesteps):
        """
        SB3-compatible training function.
        """
        self.train(total_timesteps)

    def evaluate(self, n_episodes=10):
        """
        Evaluate the agent on the environment.
        """
        rewards = []
        for _ in range(n_episodes):
            state, _ = self.env.reset()
            state = state['hero_position']
            episode_reward = 0
            done = False
            while not done:
                action = np.argmax(self.q_table[state])  # Greedy action
                state, reward, done, _, _ = self.env.step(action)
                state = state['hero_position']
                episode_reward += reward
            rewards.append(episode_reward)
        return np.mean(rewards), rewards
