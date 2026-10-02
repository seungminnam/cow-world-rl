"""
Turning an observation into a Q-table row.

This sits on the agent's side of the line on purpose. The environment reports
where things actually are; deciding that two distant cows are "the same
situation" is a modeling choice the learner makes, not a fact about the world.

Sutton & Barto frame the Markov property as a restriction on the state rather
than on the process (sec 3.1, p.49), which is a useful reminder of whose job
this is.
"""

from __future__ import annotations
import numpy as np
from cow_world import GRID


def n_states(clip: int) -> int:
    """Robot row (8) x robot col (8) x dr (span) x dc (span)."""
    span = 2 * clip + 1          # clip=2 -> offsets -2,-1,0,1,2 -> span 5
    return GRID * GRID * span * span


def encode(obs, clip: int) -> int:
    """Pack an observation into one Q-table index.

    Built like h:m:s -> seconds: each field is multiplied by how many values sit
    to its right. Cows further away than `clip` land on the same index.
    """
    robot_r, robot_c, cow_r, cow_c = (int(v) for v in obs)
    span = 2 * clip + 1

    # shift up by clip so the smallest offset is 0 -- indices can't be negative
    dr = int(np.clip(cow_r - robot_r, -clip, clip)) + clip
    dc = int(np.clip(cow_c - robot_c, -clip, clip)) + clip

    return ((robot_r * GRID + robot_c) * span + dr) * span + dc
