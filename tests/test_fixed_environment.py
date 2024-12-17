import warnings
import pytest
from gymnasium.utils.env_checker import check_env
from gymnasium import Env, make

from treasure_hunt.environment import FixedTreasureHuntEnv


@pytest.fixture
def env():
    """Fixture to initialize the environment before each test."""
    env = make("FixedTreasureHunt-v0")
    env.reset(seed=47)
    return env


def test_env_checker(env: Env):
    """Test that the environment passes Gymnasium's EnvChecker."""

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        check_env(env.unwrapped, skip_render_check=True)


def test_initialization(env: Env):
    """Test the initial state of the environment."""
    obs, _ = env.reset()
    assert obs["hero_position"] == env.FIXED_LAYOUT["hero_position"]
    assert obs["treasure_position"] == env.FIXED_LAYOUT["treasure_position"]
    assert obs["monster_positions"] == env.FIXED_LAYOUT["monster_positions"]


def test_invalid_move_penalty(env: Env):
    """Test that invalid moves (out of bounds) return the correct penalty."""
    # Test moving out of bounds (hero at position 0, try to move up)
    obs, reward, done, _, _ = env.step(0)  # action 0: up
    assert reward == env.INVALID_MOVE_PENALTY
    assert not done


def test_move_to_treasure(env: Env):
    """Test the scenario where the hero moves to the treasure."""
    # Move hero to the treasure (position 99)
    env.hero_position = 0  # Start at position 0 (top-left)
    env.treasure_position = 99  # Treasure is at the bottom-right
    # Move right 9 times and down 9 times (moving hero to position 99)
    actions = [3] * 9 + [1] * 9

    for action in actions:
        _, reward, done, _, _ = env.step(action)
        if done:
            break

    assert reward == env.TREASURE_REWARD
    assert done


def test_move_into_monster(env: Env):
    """Test the scenario where the hero moves into a monster."""
    # Place hero at a position where a monster is
    env.hero_position = 35
    env.monster_positions = [45, 55]

    # Action 2: down (hero stays at position 45)
    _, reward, done, _, _ = env.step(1)

    assert reward == env.CAUGHT_BY_MONSTER_PENALTY
    assert done


def test_valid_moves(env: Env):
    """Test that valid moves do not result in penalties."""
    last_position = 22
    env.hero_position = last_position
    for action in [0, 1, 2, 3]:  # Actions: up, down, left, right
        _obs, reward, done, _, _ = env.step(action)
        assert reward == -1  # No penalty for valid moves
        assert not done
        # Hero's position should change


def test_episode_reset(env: Env):
    """Test that the environment can be reset."""
    env.reset()
    obs = env._get_obs()
    assert obs["hero_position"] == env.FIXED_LAYOUT["hero_position"]
    assert obs["treasure_position"] == env.FIXED_LAYOUT["treasure_position"]
    assert obs["monster_positions"] == env.FIXED_LAYOUT["monster_positions"]


def test_render_output(env: Env, capsys: pytest.CaptureFixture[str]):
    """Test that render produces output (check using pytest's capsys)."""
    env.render()
    captured = capsys.readouterr()
    assert "Hero position" in captured.out
    assert "Treasure position" in captured.out
    assert "Monster positions" in captured.out


def test_invalid_action(env: Env):
    """Test that invalid action values raise an error."""
    with pytest.raises(ValueError):
        env.step(5)  # Invalid action, not in [0, 1, 2, 3]
