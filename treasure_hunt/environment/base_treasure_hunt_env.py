"""Base Implementation of a TreasureHuntEnv with logic common to all environments."""

import gymnasium as gym
from gymnasium import spaces
from gymnasium import register


class BaseTreasureHuntEnv(gym.Env):
    """Basic TreasureHuntEnvironment to get started."""
    INVALID_MOVE_PENALTY = -10
    TREASURE_REWARD = 200
    CAUGHT_BY_MONSTER_PENALTY = -50
    ENV_SIZE = 10

    metadata = {"render_modes": ["ansi"], 'render_fps': 30}

    def __init__(self, render_mode=None):
        super().__init__()

        # Define the observation space: hero, treasure, monsters
        self.observation_space = spaces.Dict({
            # Flattened 10x10 grid
            "hero_position": spaces.Discrete(self.ENV_SIZE**2),
            "treasure_position": spaces.Discrete(self.ENV_SIZE**2),
            # 2 monsters
            "monster_positions": spaces.Tuple([spaces.Discrete(self.ENV_SIZE**2)] * 2),
        })

        # Define the action space: 4 directions
        # 0: up, 1: down, 2: left, 3: right
        self.action_space = spaces.Discrete(4)

        self.hero_position = None
        self.treasure_position = None
        self.monster_positions = None

        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"Unsupported render mode {render_mode}.")
        self.render_mode = render_mode

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed, options=options)
        if seed is not None:
            # Pass the seed to the observation space for reproducibility
            self.observation_space.seed(seed)

        # Hero starts at the top-left corner
        self.hero_position = 0
        # Treasure is at the bottom-right corner
        self.treasure_position = self.observation_space["treasure_position"].n - 1

        # Initialize monster positions randomly
        self._initialize_monster_positions()

        return self._get_obs(), {}

    def _initialize_monster_positions(self):
        """Setup the monster positions.
        Unless overridden, this method initializes the monster positions randomly."""
        position_ok = False
        while not position_ok:  # Keep trying until monsters don't overlap with other objects
            self.monster_positions = tuple(
                self.observation_space["monster_positions"].sample())
            if self._is_valid_state():
                position_ok = True

        return self._get_obs(), {}

    def _is_valid_state(self):
        """Check if the current state is valid."""
        return (self.hero_position != self.treasure_position
                and self.hero_position not in self.monster_positions
                and self.treasure_position not in self.monster_positions)

    def step(self, action):
        # Handle hero movement and update monster positions dynamically
        reward, terminated, truncated = -1, False, False
        info = {}
        # Movement logic for the hero
        row, col = self._decode_position(self.hero_position)
        if action == 0:  # Move up
            if row > 0:
                row -= 1
            else:
                reward = self.INVALID_MOVE_PENALTY
        elif action == 1:  # Move down
            if row < 9:
                row += 1
            else:
                reward = self.INVALID_MOVE_PENALTY
        elif action == 2:  # Move left
            if col > 0:
                col -= 1
            else:
                reward = self.INVALID_MOVE_PENALTY
        elif action == 3:  # Move right
            if col < 9:
                col += 1
            else:
                reward = self.INVALID_MOVE_PENALTY
        else:
            raise ValueError(f"Invalid action {action}.")

        # Update the hero's position
        self.hero_position = self._encode_position(row, col)

        # Check if the hero has found the treasure
        if self.hero_position == self.treasure_position:
            reward = self.TREASURE_REWARD  # Positive reward for finding the treasure
            terminated = True  # End the episode

        # Check if the hero has ran into a monster
        elif self.hero_position in self.monster_positions:
            # Negative reward for encountering a monster
            reward = self.CAUGHT_BY_MONSTER_PENALTY
            terminated = True  # End the episode

        return self._get_obs(), reward, terminated, truncated, info

    def render(self):
        print(f"Hero position : {self._decode_position(self.hero_position)}")
        print(f"Treasure position : {
              self._decode_position(self.treasure_position)}")
        print(f"Monster positions : {', '.join(
            repr(self._decode_position(pos)) for pos in self.monster_positions)}")

    def _get_obs(self):
        """Return the current observation."""
        return {
            "hero_position": self.hero_position,
            "treasure_position": self.treasure_position,
            "monster_positions": tuple(self.monster_positions),
        }

    def _encode_position(self, row: int, col: int):
        """Convert row, col to a single integer position."""
        return row * self.ENV_SIZE + col

    def _decode_position(self, position: int):
        """Convert single integer position to row, col."""
        row, col = divmod(position, self.ENV_SIZE)
        return row, col

    def close(self):
        pass


register(
    id="BaseTreasureHunt-v0",
    entry_point="treasure_hunt.environment:BaseTreasureHuntEnv",
)
