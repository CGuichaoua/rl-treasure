"""Module for the RandomMovementStrategy class."""
from gymnasium import register

from .base_strategy import MonsterMovementStrategy


class RandomMovementStrategy(MonsterMovementStrategy):
    """Each monster moves randomly in one of the four directions or stays in place."""

    def move_monsters(self, monster_positions, hero_position, env_size, rng):
        proposed_positions = []
        for row, col in monster_positions:
            possible_moves = [
                (row - 1, col),  # Up
                (row + 1, col),  # Down
                (row, col - 1),  # Left
                (row, col + 1),  # Right
                (row, col),      # Stay in place
            ]
            # Filter out-of-bounds moves
            valid_moves = [
                (r, c) for r, c in possible_moves if 0 <= r < env_size and 0 <= c < env_size
            ]
            # Choose a random valid move
            new_row, new_col = rng.choice(valid_moves)
            proposed_positions.append((new_row, new_col))
        return proposed_positions


register(
    id="RandomMonsterTreasureHunt-v0",
    entry_point="treasure_hunt.environment:BaseTreasureHuntEnv",
    kwargs={"monster_strategy": RandomMovementStrategy()},
)
