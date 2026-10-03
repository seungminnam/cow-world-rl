"""
Checks on the state encoding, before anything is built on top of it.

An off-by-one here would not crash. Training would run and numbers would come
out, they would just be wrong, with two situations sharing one row of the table.
"""

from __future__ import annotations

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from cow_world import GRID, chebyshev
from encoding import encode, n_states


ALL_CELLS = [(r, c) for r in range(GRID) for c in range(GRID)]


def test_every_pair_lands_in_range():
    """64 x 64 is small enough to check exhaustively rather than sample."""
    for clip in (1, 2, 3):
        limit = n_states(clip)
        for robot in ALL_CELLS:
            for cow in ALL_CELLS:
                idx = encode((*robot, *cow), clip)
                assert 0 <= idx < limit, (robot, cow, clip, idx, limit)


def test_the_encoding_is_one_to_one():
    """Catches a wrong multiplier: no two situations may share a row.

    My first version checked that the largest index gets used. It doesn't --
    that would need the robot at (7,7) and the cow two rows off the board. Only
    1,156 of the 1,600 rows are reachable, which is geometry, not a bug.
    """
    for clip in (1, 2, 3):
        index_of = {}
        for robot in ALL_CELLS:
            for cow in ALL_CELLS:
                dr = max(-clip, min(clip, cow[0] - robot[0]))
                dc = max(-clip, min(clip, cow[1] - robot[1]))
                situation = (robot[0], robot[1], dr, dc)
                idx = encode((*robot, *cow), clip)
                assert index_of.setdefault(situation, idx) == idx, situation
        assert len(set(index_of.values())) == len(index_of), clip


def test_distant_cows_collapse_to_one_state():
    robot = (3, 3)
    assert encode((*robot, 7, 7), 2) == encode((*robot, 6, 6), 2)
    assert encode((*robot, 0, 0), 2) == encode((*robot, 1, 1), 2)


def test_nearby_cows_stay_distinct():
    robot = (3, 3)
    assert encode((*robot, 3, 4), 2) != encode((*robot, 4, 4), 2)
    assert encode((*robot, 2, 3), 2) != encode((*robot, 4, 3), 2)


def test_too_close_means_the_eight_surrounding_cells():
    """'Too close' is the 8 neighbors. Manhattan would miss the four diagonals."""
    robot = (3, 3)
    neighbors = [(2, 2), (2, 3), (2, 4), (3, 2), (3, 4), (4, 2), (4, 3), (4, 4)]
    assert all(chebyshev(robot, n) == 1 for n in neighbors)
    assert chebyshev(robot, (3, 5)) == 2


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"  ok  {name}")
    print("\nall encoding checks passed")
