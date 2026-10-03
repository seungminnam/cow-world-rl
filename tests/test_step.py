"""
Checks on step(), mostly on the three rules the task specification does not pin
down. Those are the ones worth testing: the rest comes straight from the task,
but these I decided, and docs/formulation.md section 4 makes claims about them.

The cow moves on its own, so a single seed proves nothing either way. Most of
these sweep a range of seeds and look at what shows up.
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

import numpy as np

from cow_world import GRID, MAX_STEPS, MOVES, CowWorld, chebyshev

ALL_CELLS = [(r, c) for r in range(GRID) for c in range(GRID)]


def bounded(pos, move):
    return (min(max(pos[0] + move[0], 0), GRID - 1),
            min(max(pos[1] + move[1], 0), GRID - 1))


def place(env, robot, cow, seed):
    """Drop the robot and cow somewhere specific after a seeded reset."""
    env.reset(seed=seed)
    env.robot, env.cow = robot, cow


def test_a_cow_that_walks_onto_the_robot_ends_the_episode():
    """My change to the spec, which only ends the episode when the robot moves
    onto the cow. Without this, parking next to the cow is free."""
    env = CowWorld()
    for seed in range(400):
        place(env, (4, 4), (4, 5), seed)
        *_, info = env.step(4)                      # stay put
        if info["outcome"] == "collision":
            return
    raise AssertionError("never collided in 400 seeds while standing still")


def test_trading_cells_counts_as_a_collision():
    """Robot steps right as the cow steps left. Neither lands on the other, but
    they went through each other getting there."""
    env = CowWorld()
    for seed in range(600):
        place(env, (4, 4), (4, 5), seed)
        *_, info = env.step(3)                      # right, into the cow's cell
        if info["outcome"] == "collision":
            return
    raise AssertionError("no collision in 600 seeds")


def test_every_adjacent_position_leaves_at_least_one_safe_action():
    """docs/formulation.md section 4 says collisions are avoidable in principle.
    I had written the opposite first, so this is the check that corrected it.

    An action is safe when its destination is neither the cow's cell nor
    anywhere the cow can reach on its own move.
    """
    worst = 5
    for robot in ALL_CELLS:
        for cow in ALL_CELLS:
            if robot == cow or chebyshev(robot, cow) != 1:
                continue
            reach = {cow} | {bounded(cow, MOVES[i]) for i in range(4)}
            safe = sum(1 for a in range(5)
                       if (d := bounded(robot, MOVES[a])) != cow and d not in reach)
            assert safe > 0, (robot, cow)
            worst = min(worst, safe)
    assert worst == 1, f"expected the corner case to leave exactly one, got {worst}"


def test_rewards_match_the_outcome():
    """Every step carries the -0.1, so the totals are one tenth below the
    headline numbers. Grouping by outcome catches a reward leaking into a case
    it does not belong to."""
    env = CowWorld()
    seen = {}
    for seed in range(300):
        place(env, (4, 4), (4, 6), seed)
        _, reward, _, _, info = env.step(3)
        key = (info["outcome"], chebyshev(env.robot, env.cow))
        seen.setdefault(key, set()).add(round(reward, 2))

    assert seen[(None, 1)] == {-3.1}, seen[(None, 1)]
    assert seen[(None, 2)] == {-0.1}, seen[(None, 2)]
    assert seen[("collision", 0)] == {-10.1}, seen[("collision", 0)]


def test_reaching_the_goal_pays_and_ends():
    env = CowWorld()
    place(env, (7, 6), (0, 0), 0)
    _, reward, terminated, _, info = env.step(3)
    assert terminated and info["outcome"] == "goal"
    assert reward == -0.1 + 20.0


def test_walking_into_the_cow_ends_it():
    env = CowWorld()
    place(env, (4, 3), (4, 4), 0)
    _, reward, terminated, _, info = env.step(3)
    assert terminated and info["outcome"] == "collision"
    assert reward == -0.1 + -10.0


def test_walls_hold():
    env = CowWorld()
    env.reset(seed=0)
    env.step(0)                                     # up, from row 0
    assert env.robot == (0, 0)


def test_the_episode_truncates_at_sixty_steps():
    env = CowWorld()
    env.reset(seed=1)
    for _ in range(MAX_STEPS):
        *_, truncated, info = env.step(4)
        if truncated:
            break
    assert truncated and env.steps == MAX_STEPS
    assert info["outcome"] == "timeout"


def test_the_same_seed_replays_the_same_episode():
    """Everything in this project is a comparison between two policies on a
    world that moves by itself, so this is the check the rest rests on."""
    env = CowWorld()
    runs = []
    for _ in range(2):
        env.reset(seed=99)
        runs.append([env.step(3)[:3] for _ in range(12)])
    for a, b in zip(*runs):
        assert np.array_equal(a[0], b[0]) and a[1:] == b[1:]


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"  ok  {name}")
    print("\nall step checks passed")
