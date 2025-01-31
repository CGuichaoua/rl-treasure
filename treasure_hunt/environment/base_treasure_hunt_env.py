"""Base Implementation of a TreasureHuntEnv with logic common to all environments."""

import gymnasium as gym
from gymnasium import spaces
from gymnasium import register
import pygame

from .monster_strategy import MonsterMovementStrategy, StationaryStrategy


class BaseTreasureHuntEnv(gym.Env):
    """Basic TreasureHuntEnvironment to get started."""
    INVALID_MOVE_PENALTY = -10  # Tried an invalid move
    TREASURE_REWARD = 200  # Won the game
    CAUGHT_BY_MONSTER_PENALTY = -50  # Lost the game
    SLACK_PENALTY = -1  # Penalty for each step to encourage fast solves
    ENV_SIZE = 10  # Constant for now, might change in the future

    metadata = {"render_modes": ["ansi", "human"], 'render_fps': 5}

    def __init__(self, render_mode=None, monster_strategy: MonsterMovementStrategy = None):
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

        if monster_strategy is not None:
            self.monster_strategy = monster_strategy
        else:
            # Default is not to move the monsters
            self.monster_strategy = StationaryStrategy()

        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"Unsupported render mode {render_mode}.")
        self.render_mode = render_mode or "ansi"

        if self.render_mode == "human":
            pygame.init()
            self.window_size = 600  # Size of the window
            self.cell_size = self.window_size // self.ENV_SIZE
            self.screen = pygame.display.set_mode(
                (self.window_size, self.window_size))
            pygame.display.set_caption("Treasure Hunt")

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
        # Takes a full turn of the hero, then the monsters
        terminated, truncated = False, False
        info = {}

        # Movement logic for the hero
        reward = self._hero_move(action)

        # Check if the hero has found the treasure
        if self.hero_position == self.treasure_position:
            reward = self.TREASURE_REWARD  # Positive reward for finding the treasure
            terminated = True  # End the episode

        # Check if the hero has ran into a monster
        elif self.hero_position in self.monster_positions:
            # Negative reward for encountering a monster
            reward = self.CAUGHT_BY_MONSTER_PENALTY
            terminated = True  # End the episode

        self._move_monsters()

        # Check if the monsters caught the hero
        if self.hero_position in self.monster_positions:
            # Negative reward for encountering a monster
            reward = self.CAUGHT_BY_MONSTER_PENALTY
            terminated = True  # End the episode

        return self._get_obs(), reward, terminated, truncated, info

    def _hero_move(self, action: int):
        row, col = self.decode_position(self.hero_position)
        if action == 0:  # Move up
            row -= 1
        elif action == 1:  # Move down
            row += 1
        elif action == 2:  # Move left
            col -= 1
        elif action == 3:  # Move right
            col += 1
        else:
            raise ValueError(f"Invalid action {action}.")
        if self._is_valid_position(row, col):
            reward = self.SLACK_PENALTY
            self.hero_position = self._encode_position(row, col)
        else:
            reward = self.INVALID_MOVE_PENALTY
        return reward

    def _move_monsters(self):
        """Moves the monsters according to strategy."""
        proposed_positions = self.monster_strategy.move_monsters(
            [self.decode_position(pos) for pos in self.monster_positions],
            self.hero_position, self.ENV_SIZE, self.np_random)
        if self._is_valid_monster_move(proposed_positions):
            # Move if the proposed monster positions are valid, stay otherwise
            self.monster_positions = tuple(
                self._encode_position(*pos) for pos in proposed_positions)

    def _is_valid_position(self, row, col):
        """Takes a proposed (decoded) position and checks if it's valid."""
        return 0 <= row < self.ENV_SIZE and 0 <= col < self.ENV_SIZE

    def render(self):
        if self.render_mode == "ansi":
            print(f"Hero position : {
                  self.decode_position(self.hero_position)}")
            print(f"Treasure position : {
                  self.decode_position(self.treasure_position)}")
            print(f"Monster positions : {', '.join(
                repr(self.decode_position(pos)) for pos in self.monster_positions)}")
        elif self.render_mode == "human":
            self._render_human()

    def _render_human(self):
        self.screen.fill((255, 255, 255))  # Fill the screen with white

        # Draw the hero
        hero_row, hero_col = self.decode_position(self.hero_position)
        pygame.draw.rect(self.screen, (0, 0, 255), (hero_col * self.cell_size,
                         hero_row * self.cell_size, self.cell_size, self.cell_size))

        # Draw the treasure
        treasure_row, treasure_col = self.decode_position(
            self.treasure_position)
        pygame.draw.rect(self.screen, (255, 215, 0), (treasure_col * self.cell_size,
                         treasure_row * self.cell_size, self.cell_size, self.cell_size))

        # Draw the monsters
        for pos in self.monster_positions:
            monster_row, monster_col = self.decode_position(pos)
            pygame.draw.rect(self.screen, (255, 0, 0), (monster_col * self.cell_size,
                             monster_row * self.cell_size, self.cell_size, self.cell_size))

        pygame.display.flip()

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

    def decode_position(self, position: int):
        """Convert single integer position to row, col."""
        row, col = divmod(position, self.ENV_SIZE)
        return row, col

    def close(self):
        if self.render_mode == "human":
            pygame.quit()

    def _is_valid_monster_move(self, proposed_positions: list[tuple[int, int]]):
        """
        Check if all proposed monster positions are valid.
        :param proposed_positions: List of new monster positions.
        :return: True if all positions are valid, False otherwise.
        """
        # Ensure the correct number of positions are provided
        if len(proposed_positions) != len(self.monster_positions):
            return False

        # Ensure positions are within bounds
        if not all(self._is_valid_position(*pos) for pos in proposed_positions):
            return False

        # Ensure no overlap with the treasure
        if self.decode_position(self.treasure_position) in proposed_positions:
            return False

        # Ensure no two monsters share the same position
        if len(proposed_positions) != len(set(proposed_positions)):
            return False

        return True


register(
    id="BaseTreasureHunt-v0",
    entry_point="treasure_hunt.environment:BaseTreasureHuntEnv",
)
