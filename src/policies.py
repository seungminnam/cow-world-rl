"""
The policies this project compares.

They all take (obs, info) and return an action, so the evaluation harness can
run any of them without knowing which it has.

The rule-based one is the bar the learned policy has to clear, and I have kept
it to what the task actually suggests: head for the goal, never step next to the
cow. It would be easy to make it smarter -- have it reason about where the cow
could move to rather than where it is -- but a baseline tuned until it is good
is a second model, and then neither number means much.
"""

from __future__ import annotations

import numpy as np

from cow_world import GRID, MOVES, chebyshev


def manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
    """Steps needed on a 4-direction grid, so the right distance to the goal.
    Chebyshev is the right one for the cow; these are different questions."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def destination(pos: tuple[int, int], action: int) -> tuple[int, int]:
    dr, dc = MOVES[action]
    return (min(max(pos[0] + dr, 0), GRID - 1), min(max(pos[1] + dc, 0), GRID - 1))


def split_obs(obs) -> tuple[tuple[int, int], tuple[int, int]]:
    r, c, cr, cc = (int(v) for v in obs)
    return (r, c), (cr, cc)


def rule_based(obs, info) -> int:
    """Move toward the goal, but never onto or beside the cow.

    It does not look ahead to where the cow might move, so it can step into a
    cell the cow then walks into. It is also memoryless, so when the cow sits on
    the short path it cannot commit to a detour and may oscillate instead. Both
    are left in.
    """
    robot, cow = split_obs(obs)
    goal = info["goal"]

    options = [(a, destination(robot, a)) for a in range(len(MOVES))]
    safe = [(a, d) for a, d in options if d != cow and chebyshev(d, cow) > 1]

    if safe:
        return min(safe, key=lambda t: (manhattan(t[1], goal), t[0]))[0]

    # Boxed in. Back off as far from the cow as possible, breaking ties toward
    # the goal. This is the case the greedy rule handles worst.
    return max(options, key=lambda t: (chebyshev(t[1], cow), -manhattan(t[1], goal)))[0]


def greedy_from_table(q_table: np.ndarray, clip: int):
    """Wrap a learned Q-table so the harness can run it like any other policy."""
    from encoding import encode

    def policy(obs, info) -> int:
        return int(np.argmax(q_table[encode(obs, clip)]))

    return policy


def random_policy(rng: np.random.Generator):
    """The floor. If a policy cannot beat this, it has not learned anything."""

    def policy(obs, info) -> int:
        return int(rng.integers(len(MOVES)))

    return policy
