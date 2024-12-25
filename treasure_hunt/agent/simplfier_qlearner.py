"""Module for the SimplifierQLearner agent class."""

from .tabular_qlearner import TabularQLearner
from .env_reducer import EnvironmentReducer


class SimplifierQLearner(TabularQLearner):
    """A Q-learner that uses an environment reducer to simplify the state space."""

    def __init__(self, env, reducer: EnvironmentReducer, *args, **kwargs):
        """Initialize the agent."""
        super().__init__(env, *args, **kwargs)
        self.reducer = reducer

    def _serialize_state(self, state):
        """Convert the observation dict to a simpler state, then to a hashable tuple."""
        state = self.reducer.reduce_observation(state)
        return super()._serialize_state(state)
