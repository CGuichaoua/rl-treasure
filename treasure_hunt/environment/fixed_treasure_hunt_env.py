"""Simple Implementation of a TreasureHuntEnv with fixed monster positions."""

from gymnasium import register

from .base_treasure_hunt_env import BaseTreasureHuntEnv


class FixedTreasureHuntEnv(BaseTreasureHuntEnv):
    """Simple TreasureHuntEnv to get started. 
    Treasure and Monster positions are fixed and monsters don't move.
    """

    FIXED_LAYOUT = {
        "hero_position": 0,
        "treasure_position": 99,
        "monster_positions": (45, 55),
    }

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed, options=options)

        # Use the fixed layout
        self.hero_position = self.FIXED_LAYOUT["hero_position"]
        self.treasure_position = self.FIXED_LAYOUT["treasure_position"]
        self.monster_positions = self.FIXED_LAYOUT["monster_positions"]

        return self._get_obs(), {}


register(
    id="FixedTreasureHunt-v0",
    entry_point="treasure_hunt.environment:FixedTreasureHuntEnv",
)
