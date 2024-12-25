"""This module contains the stationary strategy for the monster movement."""

from .base_strategy import MonsterMovementStrategy


class StationaryStrategy(MonsterMovementStrategy):
    """Do not move."""

    def move_monsters(self, monster_positions, hero_position, env_size, rng):
        return monster_positions
