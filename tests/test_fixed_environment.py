"""Tests for the FixedTreasureHuntEnv environment."""
import warnings
import pytest
from gymnasium.utils.env_checker import check_env

from treasure_hunt.environment import FixedTreasureHuntEnv

# pylint: disable=W0611 # Unused import
from .fixtures import fixture_environment


def test_env_checker(environment: FixedTreasureHuntEnv):
    """Test that the environment passes Gymnasium's EnvChecker."""

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        check_env(environment.unwrapped, skip_render_check=True)


def test_initialization(environment: FixedTreasureHuntEnv):
    """Test the initial state of the environment."""
    obs, _ = environment.reset()
    assert obs["hero_position"] == environment.FIXED_LAYOUT["hero_position"]
    assert obs["treasure_position"] == environment.FIXED_LAYOUT["treasure_position"]
    assert obs["monster_positions"] == environment.FIXED_LAYOUT["monster_positions"]


def test_invalid_move_penalty(environment: FixedTreasureHuntEnv):
    """Test that invalid moves (out of bounds) return the correct penalty."""
    # Test moving out of bounds (hero at position 0, try to move up)
    _obs, reward, done, _, _ = environment.step(0)  # action 0: up
    assert reward == environment.INVALID_MOVE_PENALTY
    assert not done


def test_move_to_treasure(environment: FixedTreasureHuntEnv):
    """Test the scenario where the hero moves to the treasure."""
    # Move hero to the treasure (position 99)
    environment.hero_position = 0  # Start at position 0 (top-left)
    environment.treasure_position = 99  # Treasure is at the bottom-right
    # Move right 9 times and down 9 times (moving hero to position 99)
    actions = [3] * 9 + [1] * 9

    for action in actions:
        _, reward, done, _, _ = environment.step(action)
        if done:
            break

    assert reward == environment.TREASURE_REWARD
    assert done


def test_move_into_monster(environment: FixedTreasureHuntEnv):
    """Test the scenario where the hero moves into a monster."""
    # Place hero at a position where a monster is
    environment.hero_position = 35
    environment.monster_positions = [45, 55]

    # Action 2: down (hero stays at position 45)
    _, reward, done, _, _ = environment.step(1)

    assert reward == environment.CAUGHT_BY_MONSTER_PENALTY
    assert done


def test_valid_moves(environment: FixedTreasureHuntEnv):
    """Test that valid moves do not result in penalties."""
    last_position = 22
    environment.hero_position = last_position
    for action in [0, 1, 2, 3]:  # Actions: up, down, left, right
        _obs, reward, done, _, _ = environment.step(action)
        assert reward == -1  # No penalty for valid moves
        assert not done
        # Hero's position should change


def test_episode_reset(environment: FixedTreasureHuntEnv):
    """Test that the environment can be reset."""
    obs, _ = environment.reset()
    assert obs["hero_position"] == environment.FIXED_LAYOUT["hero_position"]
    assert obs["treasure_position"] == environment.FIXED_LAYOUT["treasure_position"]
    assert obs["monster_positions"] == environment.FIXED_LAYOUT["monster_positions"]


def test_render_output(environment: FixedTreasureHuntEnv, capsys: pytest.CaptureFixture[str]):
    """Test that render produces output (check using pytest's capsys)."""
    environment.render()
    captured = capsys.readouterr()
    assert "Hero position" in captured.out
    assert "Treasure position" in captured.out
    assert "Monster positions" in captured.out


def test_invalid_action(environment: FixedTreasureHuntEnv):
    """Test that invalid action values raise an error."""
    with pytest.raises(ValueError):
        environment.step(5)  # Invalid action, not in [0, 1, 2, 3]
