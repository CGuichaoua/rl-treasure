"""Shared fixtures for tests."""

import pytest

from gymnasium.envs import make

from treasure_hunt.environment import FixedTreasureHuntEnv
from treasure_hunt.agent import TabularQLearner


@pytest.fixture(name='environment')
def fixture_environment():
    """Fixture to initialize the environment before each test."""
    env = make("FixedTreasureHunt-v0")
    env.reset(seed=47)
    return env.unwrapped


@pytest.fixture(name="q_learner")
def fixture_q_learner(environment: FixedTreasureHuntEnv):
    """Fixture to create the TabularQLearner."""
    return TabularQLearner(environment)
