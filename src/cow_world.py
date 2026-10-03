"""
An 8x8 grid world where a robot has to reach a goal without hitting a wandering cow.

Why the state looks the way it does is in docs/formulation.md.
"""

from __future__ import annotations
import numpy as np
import gymnasium as gym
from gymnasium import spaces

# --- world -----------------------------------------------------------------

GRID = 8              # 8 x 8 cells
MAX_STEPS = 60        # the episode truncates here
COW_MOVE_PROB = 0.7   # chance the cow moves at all on a given step

# --- rewards ---------------------------------------------------------------

R_GOAL = 20.0
R_COLLISION = -10.0
R_TOO_CLOSE = -3.0    # cow is in one of the 8 cells around the robot
R_STEP = -0.1

# --- actions ---------------------------------------------------------------
# The list index is the action number, so MOVES[2] is what action 2 does.
MOVES = [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]
ACTION_NAMES = ["up", "down", "left", "right", "stay"]


def chebyshev(a: tuple[int, int], b: tuple[int, int]) -> int:
    """King-move distance. Distance 1 is the 8 surrounding cells, i.e. 'too close'."""
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


class CowWorld(gym.Env):
    """Robot starts at (0,0) and has to reach (7,7). A cow starts at the center
    and wanders.

    The observation is the true state. What a learner does with it -- including
    throwing detail away -- is the learner's business, so that lives in
    encoding.py rather than here.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode: str | None = None):
        super().__init__()

        # whatever reset() and step() return has to fit these
        self.observation_space = spaces.MultiDiscrete([GRID, GRID, GRID, GRID])
        self.action_space = spaces.Discrete(len(MOVES))
        self.render_mode = render_mode

        self.start = (0, 0)
        self.goal = (GRID - 1, GRID - 1)
        self.cow_start = (GRID // 2, GRID // 2)

        self.robot = self.start
        self.cow = self.cow_start
        self.steps = 0

    # --- helpers -----------------------------------------------------------

    def _obs(self) -> np.ndarray:
        return np.array([*self.robot, *self.cow], dtype=np.int64)

    def _info(self) -> dict:
        """Bookkeeping that does not belong in the observation. Positions come
        through the observation now."""
        return {
            "goal": self.goal,
            "steps": self.steps,
            "too_close": chebyshev(self.robot, self.cow) == 1,
        }

    # --- gym API -----------------------------------------------------------

    def reset(self, seed: int | None = None, options: dict | None = None):
        # seeds self.np_random, so the same seed replays the same cow
        super().reset(seed=seed)

        self.robot = self.start
        self.cow = self.cow_start
        self.steps = 0
        return self._obs(), self._info()
