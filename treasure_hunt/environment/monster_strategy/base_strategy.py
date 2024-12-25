"""Interface for monster movement strategies."""

from abc import ABC, abstractmethod


class MonsterMovementStrategy(ABC):
    """Abstract base class for monster movement strategies."""

    @abstractmethod
    def move_monsters(self, monster_positions: list[tuple[int, int]], hero_position: tuple[int, int], env_size, rng) -> bool:
        """
        Move all monsters in the environment.
        :param env: Size of the grid (e.g., 10x10).
        :param monster_positions: Current position of the monsters.
        :param hero_position: Current position of the hero.
        :param env_size: Size of the grid (e.g., 10x10).
        :return: New position of the monster.
        """
