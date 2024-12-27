
import gymnasium as gym
from gymnasium import ObservationWrapper
import numpy as np


class FlattenTreasureWrapper(ObservationWrapper):
    """
    A wrapper that flattens the observation space of the environment.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)

        # Define the new observation space
        self.observation_space = gym.spaces.MultiDiscrete(
            [env.observation_space.spaces['hero_position'].n,
                env.observation_space.spaces['treasure_position'].n,
                * (space.n for space in env.observation_space.spaces['monster_positions'])
             ])

    def observation(self, observation):
        """
        Flatten the observation space.
        """
        return np.array([observation['hero_position'], observation['treasure_position'], *observation['monster_positions']])
