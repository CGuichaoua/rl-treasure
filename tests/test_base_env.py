"""Tests for the TreasureHuntEnv environment."""
import warnings
import pytest
from gymnasium.utils.env_checker import check_env

from treasure_hunt.environment import BaseTreasureHuntEnv

# pylint: disable=W0212  # We're fine with using protected members in tests.
# pylint: disable=W0611 # Unused import
from .fixtures import fixture_base_environment


class TestBaseTreasureHuntEnv:
    """Tests common to all TreasureHuntEnv environments."""
    @pytest.fixture
    def environment(self, base_environment) -> BaseTreasureHuntEnv:
        """Fixture to get the right type of environment."""
        return base_environment

    def test_env_checker(self, environment):
        """Test that the environment passes Gymnasium's EnvChecker."""

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            check_env(environment.unwrapped, skip_render_check=True)

    def test_initialization(self, environment):
        """Test the initial state of the environment."""
        obs, _ = environment.reset()
        assert obs["hero_position"] == 0
        assert obs["treasure_position"] == environment.observation_space["treasure_position"].n - 1
        assert len(set(obs["monster_positions"]) & {
                   obs["hero_position"], obs["treasure_position"]}) == 0

    def test_invalid_move_penalty(self, environment):
        """Test that invalid moves (out of bounds) return the correct penalty."""
        # Test moving out of bounds (hero at position 0, try to move up)
        _obs, reward, done, _, _ = environment.step(0)  # action 0: up
        assert reward == environment.INVALID_MOVE_PENALTY
        assert not done

    def test_move_to_treasure(self, environment):
        """Test the scenario where the hero moves to the treasure."""
        # Cheat the hero next to the treasure
        environment.hero_position = 89
        # Move the hero to the treasure (position 99)
        _, reward, done, _, _ = environment.step(1)  # action 1: down

        assert reward == environment.TREASURE_REWARD
        assert done

    def test_move_into_monster(self, environment):
        """Test the scenario where the hero moves into a monster."""
        # Place hero at a position where a monster is
        environment.hero_position = 35
        environment.monster_positions = [45, 55]

        # Action 2: down
        _, reward, done, _, _ = environment.step(1)

        assert reward == environment.CAUGHT_BY_MONSTER_PENALTY
        assert done

    def test_valid_moves(self, environment):
        """Test that valid moves do not result in penalties."""
        # Place monsters out of hero's way
        environment.monster_positions = [89, 88]
        environment.hero_position = 22
        for action in [0, 1, 2, 3]:  # Actions: up, down, left, right
            _obs, reward, done, _, _ = environment.step(action)
            assert reward == -1  # No penalty for valid moves
            assert not done
            # Hero's position should change

    def test_episode_reset(self, environment):
        """Test that the environment can be reset."""
        obs, _ = environment.reset()
        assert obs["hero_position"] == 0
        assert obs["treasure_position"] == environment.observation_space["treasure_position"].n - 1
        assert len(set(obs["monster_positions"]) & {
                   obs["hero_position"], obs["treasure_position"]}) == 0

    def test_render_output(self, environment, capsys: pytest.CaptureFixture[str]):
        """Test that render produces output (check using pytest's capsys)."""
        environment.render()
        captured = capsys.readouterr()
        assert "Hero position" in captured.out
        assert "Treasure position" in captured.out
        assert "Monster positions" in captured.out

    def test_invalid_action(self, environment):
        """Test that invalid action values raise an error."""
        with pytest.raises(ValueError):
            environment.step(5)  # Invalid action, not in [0, 1, 2, 3]

    def test_is_valid_state(self, environment):
        """Test the _is_valid_state method."""
        environment.hero_position = 0
        environment.treasure_position = 99
        environment.monster_positions = [10, 20]
        assert environment._is_valid_state()

        environment.monster_positions = [0, 20]
        assert not environment._is_valid_state()

        environment.monster_positions = [10, 99]
        assert not environment._is_valid_state()

    def test_is_valid_position(self, environment):
        """Test the _is_valid_position method."""
        assert environment._is_valid_position(0, 0)
        assert environment._is_valid_position(9, 9)
        assert not environment._is_valid_position(-1, 0)
        assert not environment._is_valid_position(0, 10)

    def test_is_valid_monster_move(self, environment):
        """Test the _is_valid_monster_move method."""
        environment.treasure_position = 99
        assert environment._is_valid_monster_move([(1, 1), (2, 2)])
        assert not environment._is_valid_monster_move([(1, 1), (1, 1)])
        assert not environment._is_valid_monster_move([(1, 1), (99, 99)])
        assert not environment._is_valid_monster_move([(1, 1)])
