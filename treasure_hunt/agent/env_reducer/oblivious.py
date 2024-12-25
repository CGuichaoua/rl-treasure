"""Module for the ObliviousReducer environment reducer class."""

from .environment_reducer import EnvironmentReducer


class ObliviousReducer(EnvironmentReducer):
    """Ignore all monsters."""

    def __init__(self, env, dropped_feature: str = "monster_positions"):
        """Initialize the reducer."""
        super().__init__(env)
        self.dropped_feature = dropped_feature

    def reduce_observation(self, obs):
        """Return only the hero and treasure positions."""
        return {k: v for k, v in obs.items() if k != self.dropped_feature}
