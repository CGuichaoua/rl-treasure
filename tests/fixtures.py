"""Shared fixtures for tests."""

import pytest

from gymnasium.envs import make
from gymnasium.wrappers import TimeLimit

from treasure_hunt.environment import FixedTreasureHuntEnv
from treasure_hunt.agent import TabularQLearner


@pytest.fixture(name='fixed_environment')
def fixture_fixed_environment():
    """Fixture to initialize a fixed environment before each test."""
    env = make("FixedTreasureHunt-v0")
    env.reset(seed=47)
    return env.unwrapped


@pytest.fixture(name='base_environment')
def fixture_base_environment():
    """Fixture to initialize a base environment before each test."""
    env = make("BaseTreasureHunt-v0")
    env.reset(seed=47)
    return env.unwrapped


@pytest.fixture(name="q_learner")
def fixture_q_learner(environment: FixedTreasureHuntEnv):
    """Fixture to create the TabularQLearner."""
    return TabularQLearner(environment)


@pytest.fixture(name="q_learner_with_limit")
def fixture_q_learner_with_limit(environment: FixedTreasureHuntEnv):
    """Fixture to create the TabularQLearner with a limit."""
    limited_env = TimeLimit(environment, max_episode_steps=500)
    return TabularQLearner(limited_env)
