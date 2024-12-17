"""Basic Implementation of a TreasureHuntEnv."""

import gymnasium as gym
from gymnasium import spaces
from gymnasium import register


class FixedTreasureHuntEnv(gym.Env):
    """Basic TreasureHuntEnvironment to get started. Treasure and Monster positions are fixed and monsters don't move."""
    FIXED_LAYOUT = {
        "hero_position": 0,
        "treasure_position": 99,
        "monster_positions": [45, 55],
    }
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

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)

        # Use the fixed layout
        self.hero_position = self.FIXED_LAYOUT["hero_position"]
        self.treasure_position = self.FIXED_LAYOUT["treasure_position"]
        self.monster_positions = self.FIXED_LAYOUT["monster_positions"]

        return self._get_obs(), {}

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
        return {
            "hero_position": self.hero_position,
            "treasure_position": self.treasure_position,
            "monster_positions": tuple(self.monster_positions),
        }

    def _encode_position(self, row: int, col: int):
        return row * self.ENV_SIZE + col

    def _decode_position(self, position: int):
        row, col = divmod(position, self.ENV_SIZE)
        return row, col

    def close(self):
        pass


register(
    id="FixedTreasureHunt-v0",
    entry_point="treasure_hunt.environment:FixedTreasureHuntEnv",
)
