"""Tests for the StationaryStragy class"""
import pytest
from treasure_hunt.environment.monster_strategy.stationary import StationaryStrategy


class TestStationaryStrategy:
    """Tests for the StationaryStrategy class."""

    @pytest.fixture
    def strategy(self):
        """Fixture to create an instance of StationaryStrategy."""
        return StationaryStrategy()

    @pytest.mark.parametrize("initial_positions, hero_position, env_size", [
        ([(1, 1), (2, 2), (3, 3)], (0, 0), 10),
        ([], (0, 0), 10),
        ([(5, 5)], (0, 0), 10),
        ([(10, 10), (20, 20), (30, 30), (40, 40), (50, 50)], (0, 0), 100)
    ])
    def test_immobile_monsters(self, strategy, initial_positions, hero_position, env_size):
        """Test that monsters do not move."""
        rng = None  # Random number generator is not used in this strategy
        new_positions = strategy.move_monsters(
            initial_positions, hero_position, env_size, rng)
        assert new_positions == initial_positions, "Monsters should not move in StationaryStrategy."
        assert isinstance(new_positions, list), "Expected a list of tuples."
        assert all(isinstance(pos, tuple)
                   for pos in new_positions), "Expected a list of tuples."
