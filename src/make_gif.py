"""
Side-by-side episodes for the two policies.

Both panels run the same seed, so the cow does the same thing in each and any
difference on screen comes from the robot. Running them separately would have
shown two different cows and proved nothing.

    python src/make_gif.py --seeds 1000003 1000011 1000042 --out results/episodes.gif
"""

from __future__ import annotations

import argparse
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from cow_world import GRID, MAX_STEPS, CowWorld, chebyshev
from policies import greedy_from_table, rule_based

ROOT = pathlib.Path(__file__).resolve().parent.parent
BLUE, BROWN, GREEN, RING = "#1f6feb", "#8b5a2b", "#2e8b57", "#f6d7d7"


def trace(policy, seed: int) -> tuple[list, str]:
    """Replay one episode and keep every (robot, cow) pair along the way."""
    env = CowWorld()
    obs, info = env.reset(seed=seed)
    frames = [(env.robot, env.cow, 0.0)]
    total = 0.0
    outcome = "timeout"

    for _ in range(MAX_STEPS):
        obs, reward, terminated, truncated, info = env.step(policy(obs, info))
        total += reward
        frames.append((env.robot, env.cow, total))
        if terminated or truncated:
            outcome = info.get("outcome") or "timeout"
            break

    return frames, outcome


def draw(ax, robot, cow, goal, title: str):
    ax.clear()
    ax.set_xlim(-0.5, GRID - 0.5)
    ax.set_ylim(GRID - 0.5, -0.5)
    ax.set_xticks(range(GRID))
    ax.set_yticks(range(GRID))
    ax.set_aspect("equal")
    ax.grid(True, color="#dddddd", lw=0.6)
    ax.tick_params(labelsize=6)

    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == dc == 0:
                continue
            r, c = cow[0] + dr, cow[1] + dc
            if 0 <= r < GRID and 0 <= c < GRID:
                ax.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1, color=RING, zorder=0))

    ax.scatter(goal[1], goal[0], marker="*", s=300, color=GREEN, zorder=3)
    ax.scatter(cow[1], cow[0], marker="s", s=170, color=BROWN, zorder=3)
    ax.scatter(robot[1], robot[0], marker="o", s=170, color=BLUE, zorder=4)
    ax.set_title(title, fontsize=9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[1000003, 1000011, 1000042])
    ap.add_argument("--out", default=str(ROOT / "results" / "episodes.gif"))
    ap.add_argument("--fps", type=int, default=4)
    ap.add_argument("--clip", type=int, default=2)
    args = ap.parse_args()

    import imageio.v2 as imageio

    q = np.load(ROOT / "results" / f"q_base.npy")
    learned = greedy_from_table(q, args.clip)
    goal = (GRID - 1, GRID - 1)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.9))
    images = []

    for seed in args.seeds:
        left, left_outcome = trace(rule_based, seed)
        right, right_outcome = trace(learned, seed)
        length = max(len(left), len(right))

        for i in range(length):
            lr, lc, lret = left[min(i, len(left) - 1)]
            rr, rc, rret = right[min(i, len(right) - 1)]
            ldone = "  " + left_outcome if i >= len(left) - 1 else ""
            rdone = "  " + right_outcome if i >= len(right) - 1 else ""

            draw(axes[0], lr, lc, goal, f"rule-based   step {min(i, len(left)-1)}{ldone}")
            draw(axes[1], rr, rc, goal, f"Q-learning   step {min(i, len(right)-1)}{rdone}")
            fig.suptitle(f"seed {seed}", fontsize=10, y=0.98)
            fig.tight_layout(rect=(0, 0, 1, 0.95))
            fig.canvas.draw()
            images.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())

        # hold on the final frame so the outcome is readable
        images.extend([images[-1]] * args.fps)

        print(f"  seed {seed}: rule-based {left_outcome} in {len(left)-1} steps, "
              f"Q-learning {right_outcome} in {len(right)-1} steps")

    imageio.mimsave(args.out, images, fps=args.fps, loop=0)
    plt.close(fig)
    seconds = len(images) / args.fps
    print(f"\nwrote {args.out}  ({len(images)} frames, {seconds:.0f}s)")


if __name__ == "__main__":
    main()
