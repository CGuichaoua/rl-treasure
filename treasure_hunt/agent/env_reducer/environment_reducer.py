"""Module for the abstract EnvironmentReducer class."""
from abc import ABC, abstractmethod

from ...environment import BaseTreasureHuntEnv


class EnvironmentReducer(ABC):
    """An abstract class for reducing the environment state to a smaller state space."""

    def __init__(self, env: BaseTreasureHuntEnv):
        self.env = env

    @abstractmethod
    def reduce_observation(self, obs):
        """Turn an observation into a simpler one."""
