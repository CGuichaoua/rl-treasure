"""Tests for the FixedTreasureHuntEnv environment."""
import pytest
from treasure_hunt.environment import FixedTreasureHuntEnv
from .test_base_env import TestBaseTreasureHuntEnv

# pylint: disable=W0611  # Unused import
from .fixtures import fixture_fixed_environment, fixture_base_environment


class TestFixedTreasureHuntEnv(TestBaseTreasureHuntEnv):
    """Tests specific to the FixedTreasureHuntEnv environment."""

    @pytest.fixture
    # pylint: disable=W0237  # We're overriding a fixture from a parent class.
    def environment(self, fixed_environment) -> FixedTreasureHuntEnv:
        return fixed_environment

    def test_initialization(self, environment):
        """Test the initial state of the environment."""
        obs, _ = environment.reset()
        assert obs["hero_position"] == environment.FIXED_LAYOUT["hero_position"]
        assert obs["treasure_position"] == environment.FIXED_LAYOUT["treasure_position"]
        assert obs["monster_positions"] == environment.FIXED_LAYOUT["monster_positions"]

    def test_episode_reset(self, environment):
        """Test that the environment can be reset."""
        obs, _ = environment.reset()
        assert obs["hero_position"] == environment.FIXED_LAYOUT["hero_position"]
        assert obs["treasure_position"] == environment.FIXED_LAYOUT["treasure_position"]
        assert obs["monster_positions"] == environment.FIXED_LAYOUT["monster_positions"]
