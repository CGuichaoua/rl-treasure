"""Tests for the TabularQLearner class."""
from pathlib import Path
import numpy as np

from treasure_hunt.environment import FixedTreasureHuntEnv
from treasure_hunt.agent import TabularQLearner

# pylint: disable=W0212  # We're fine with using protected members in tests.
# pylint: disable=W0611  # Unused import
from .fixtures import fixture_fixed_environment, fixture_q_learner, fixture_q_learner_with_limit


def test_serialize_state(q_learner: TabularQLearner):
    """Test state serialization into a hashable tuple."""
    state = {
        "hero_position": 0,
        "treasure_position": 99,
        "monster_positions": (45, 55),
    }
    serialized = q_learner._serialize_state(state)
    expected = (0, 99, (45, 55))
    assert serialized == expected, f"Serialized state {
        serialized} does not match expected {expected}."


def test_select_action(q_learner: TabularQLearner):
    """Test action selection logic (exploration vs. exploitation)."""
    state = (0, 99, (45, 55))
    q_learner.q_table[state] = np.array([1, 0, 0, 0])

    # Test deterministic action selection (exploitation)
    deterministic_action = q_learner._select_action(state, deterministic=True)
    assert deterministic_action == 0, "Deterministic action should select the highest Q-value."

    # Test random action selection (exploration)
    q_learner.exploration_rate = 1.0
    actions = [q_learner._select_action(
        state, deterministic=False) for _ in range(100)]
    assert len(set(actions)) > 1, "Exploration should result in random actions."


def test_update_q_value(q_learner: TabularQLearner):
    """Test Q-value update logic."""
    state = (0, 99, (45, 55))
    next_state = (1, 99, (45, 55))
    action = 0
    reward = 10
    q_learner.q_table[state] = np.array([0, 0, 0, 0], dtype='float')
    q_learner.q_table[next_state] = np.array([5, 0, 0, 0], dtype='float')

    q_learner._update_q_value(state, action, reward, next_state)
    expected_q_value = 0 + q_learner.learning_rate * (
        reward + q_learner.discount_factor * 5 - 0)
    assert np.isclose(q_learner.q_table[state][action], expected_q_value), (
        "Q-value update does not match expected value."
    )


def test_train(q_learner: TabularQLearner):
    """Test the training process."""
    q_learner.learn(total_timesteps=10)

    # Ensure some Q-values have been updated
    non_default_states = [state for state in q_learner.q_table if np.any(
        q_learner.q_table[state] != 0)]
    assert len(
        non_default_states) > 0, "Training should update Q-values for encountered states."


def test_predict(q_learner: TabularQLearner, fixed_environment: FixedTreasureHuntEnv):
    """Test the predict method for action selection."""
    state, _ = fixed_environment.reset()
    action, _ = q_learner.predict(state, deterministic=True)

    assert action in range(
        fixed_environment.action_space.n), "Predicted action should be within the valid action space."


def test_save_and_load(q_learner: TabularQLearner, tmp_path: Path):
    """Test saving and loading the Q-table."""
    state = (0, 99, (45, 55))
    q_learner.q_table[state] = np.array([1, 2, 3, 4])

    # Save Q-table
    save_path = tmp_path / "q_table.npy"
    q_learner.save(save_path)

    # Create a new agent and load Q-table
    new_agent = TabularQLearner(q_learner.env)
    new_agent.load(save_path)

    # Verify loaded Q-table matches the original
    assert np.array_equal(new_agent.q_table[state], q_learner.q_table[state]), (
        "Loaded Q-table does not match the saved Q-table."
    )
