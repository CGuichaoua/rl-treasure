"""Module for the ObliviousReducer environment reducer class."""

from .environment_reducer import EnvironmentReducer


class NearSightedReducer(EnvironmentReducer):
    """Reduce monster position to a relative near-far state."""

    def __init__(self, env, focus_distance: int = 2):
        """Initialize the reducer."""
        super().__init__(env)
        self.focus_distance = focus_distance

    def reduce_observation(self, obs):
        """Return only the hero and treasure positions."""
        obs = obs.copy()
        obs['monster_positions'] = tuple(self._discretize_monster_position(
            monster_pos, obs['hero_position']) for monster_pos in obs['monster_positions'])
        return obs

    def _discretize_monster_position(self, monster_pos, hero_pos):
        """Discretize the monster position."""
        return tuple(self._discretize_relative_coordinate(coord) for coord in self._relative_monster_position(monster_pos, hero_pos))

    def _relative_monster_position(self, monster_pos, hero_pos):
        """Return the relative position of the monster to the hero."""
        monster_row, monster_col = self.env.decode_position(monster_pos)
        hero_row, hero_col = self.env.decode_position(hero_pos)
        row_diff = monster_row - hero_row
        col_diff = monster_col - hero_col
        return row_diff, col_diff

    def _discretize_relative_coordinate(self, coord: int):
        """Discretize a relative coordinate."""
        if coord < - self.focus_distance:
            return -1
        elif coord > self.focus_distance:
            return 1
        else:
            return 0
