"""This module contains the stationary strategy for the monster movement."""
from gymnasium import register

from .base_strategy import MonsterMovementStrategy


class StationaryStrategy(MonsterMovementStrategy):
    """Do not move."""

    def move_monsters(self, monster_positions, hero_position, env_size, rng):
        return monster_positions


register(
    id="StationaryMonsterTreasureHunt-v0",
    entry_point="treasure_hunt.environment:BaseTreasureHuntEnv",
    kwargs={"monster_strategy": StationaryStrategy()},
)
