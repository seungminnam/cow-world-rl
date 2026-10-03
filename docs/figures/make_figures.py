"""Figures for docs/formulation.md.  python docs/figures/make_figures.py"""
import pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch

OUT = pathlib.Path(__file__).resolve().parent
GRID, CLIP, ROBOT = 8, 2, (3, 3)
BLUE, BROWN, MERGE, EDGE = "#1f6feb", "#8b5a2b", "#e8eef7", "#d5d5d5"


def groups(origin, clip=CLIP, n=GRID):
    """Cells sharing a clipped offset from `origin`, as contiguous index runs."""
    out, start = [], 0
    for i in range(1, n + 1):
        if i == n or max(-clip, min(clip, i - origin)) != max(-clip, min(clip, start - origin)):
            out.append(list(range(start, i)))
            start = i
    return out


# --- Figure 1: the 8x8 grid is really only 25 states ------------------------
fig, ax = plt.subplots(figsize=(5.4, 5.4))
rows, cols = groups(ROBOT[0]), groups(ROBOT[1])

ax.set_xlim(-0.5, GRID - 0.5); ax.set_ylim(GRID - 0.5, -0.5)
ax.set_xticks(range(GRID)); ax.set_yticks(range(GRID))
ax.set_aspect("equal"); ax.tick_params(labelsize=7)
ax.grid(True, color=EDGE, lw=0.5, zorder=0)

for rg in rows:
    for cg in cols:
        if len(rg) * len(cg) > 1:          # more than one cell -> merged
            ax.add_patch(Rectangle((cg[0] - 0.5, rg[0] - 0.5), len(cg), len(rg),
                                   facecolor=MERGE, zorder=0))
        ax.add_patch(Rectangle((cg[0] - 0.5, rg[0] - 0.5), len(cg), len(rg),
                               fill=False, edgecolor="#333", lw=1.7, zorder=2))

ax.scatter(ROBOT[1], ROBOT[0], marker="o", s=270, color=BLUE, zorder=4)
for cow in [(3, 5), (3, 6), (3, 7)]:
    ax.scatter(cow[1], cow[0], marker="s", s=200, color=BROWN, zorder=4)

ax.set_title("One bordered region = one state", fontsize=11.5, pad=9)
ax.set_xlabel("robot at (3,3).  64 cells collapse to 25 states.\n"
              "the three cows sit in one region, so the agent cannot tell them apart.",
              fontsize=8.5, color="#333", labelpad=8)
fig.tight_layout()
fig.savefig(OUT / "clip_merges_states.png", dpi=160, bbox_inches="tight")

# --- Figure 2: why the radius is 2 ------------------------------------------
fig, axes = plt.subplots(2, 1, figsize=(6.2, 3.2))
for ax, gap, verdict, color in [
    (axes[0], 2, "distance 2  →  both land on cell 1 = COLLISION", "#c0392b"),
    (axes[1], 3, "distance 3  →  end up adjacent = −3, survivable", "#2e8b57"),
]:
    n = gap + 2
    ax.set_xlim(-0.6, n - 0.4); ax.set_ylim(-0.75, 0.75)
    ax.set_xticks(range(n)); ax.set_yticks([])
    ax.set_aspect("equal"); ax.grid(True, axis="x", color=EDGE, lw=0.6)
    for s in ("left", "right", "top"):
        ax.spines[s].set_visible(False)

    ax.scatter(0, 0.22, marker="o", s=165, color=BLUE, zorder=3)
    ax.scatter(gap, 0.22, marker="s", s=150, color=BROWN, zorder=3)
    ax.add_patch(FancyArrowPatch((0.12, 0.22), (0.88, -0.22), arrowstyle="-|>",
                                 mutation_scale=10, color=BLUE, lw=1.3, zorder=3))
    ax.add_patch(FancyArrowPatch((gap - 0.12, 0.22), (gap - 0.88, -0.22), arrowstyle="-|>",
                                 mutation_scale=10, color=BROWN, lw=1.3, zorder=3))
    # where they end up
    ax.scatter(1, -0.3, marker="o", s=165, facecolor="none", edgecolor=BLUE, lw=1.4, zorder=3)
    ax.scatter(gap - 1, -0.3, marker="s", s=150, facecolor="none", edgecolor=BROWN, lw=1.4, zorder=3)
    ax.set_xlabel(verdict, fontsize=9, color=color, labelpad=3)

axes[0].set_title("Robot and cow each move one cell, so one step closes two",
                  fontsize=10.5, pad=8)
fig.tight_layout()
fig.savefig(OUT / "why_radius_two.png", dpi=160, bbox_inches="tight")

# --- Figure 3: the offset range, and what clipping leaves of it ------------
fig, axes = plt.subplots(2, 1, figsize=(6.8, 2.7))

for ax, hi, title in [
    (axes[0], 7, "offset = cow − robot, on an axis of 8 cells (indices 0–7)"),
    (axes[1], CLIP, "after clipping to ±2"),
]:
    vals = list(range(-hi, hi + 1))
    ax.set_xlim(-8.3, 8.3); ax.set_ylim(-0.45, 0.62)
    ax.set_yticks([]); ax.set_xticks(vals); ax.tick_params(labelsize=8)
    for s in ("left", "right", "top"):
        ax.spines[s].set_visible(False)
    ax.scatter(vals, [0] * len(vals), s=44, color=BLUE, zorder=3)
    ax.set_xlabel(f"{len(vals)} values  =  {hi} + 1 + {hi}",
                  fontsize=8.5, color="#333", labelpad=3)
    ax.set_title(title, fontsize=9.5, pad=5)

# everything past the clip folds onto the edge value
for sign in (-1, 1):
    for v in range(CLIP + 1, 8):
        axes[1].scatter([sign * v], [0.36], s=24, color="#bbb", zorder=2)
        axes[1].add_patch(FancyArrowPatch((sign * v, 0.33), (sign * CLIP, 0.07),
                                          arrowstyle="-|>", mutation_scale=8,
                                          color="#aaa", lw=0.9,
                                          connectionstyle=f"arc3,rad={-0.22 * sign}",
                                          zorder=2))

fig.tight_layout()
fig.savefig(OUT / "offset_range.png", dpi=160, bbox_inches="tight")

# --- Figure 4: why position alone is enough (and when it would not be) -----
fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.5))
OUTGOING = [(0, -1), (0, 1), (-1, 0), (1, 0)]

for ax, weights, title, sub in [
    (axes[0], [0.25] * 4, "Our cow: a fresh roll every step",
     "however it got here, the next step is the same"),
    (axes[1], [0.05, 0.85, 0.05, 0.05], "A cow with momentum (not our cow)",
     "same cell, different future — you would need its heading"),
]:
    ax.set_xlim(-2.2, 2.2); ax.set_ylim(2.2, -2.2)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
    for s in ax.spines.values():
        s.set_visible(False)
    for g in (-1.5, -0.5, 0.5, 1.5):
        ax.axhline(g, color=EDGE, lw=0.6); ax.axvline(g, color=EDGE, lw=0.6)

    for (dr, dc), w in zip(OUTGOING, weights):
        ax.add_patch(FancyArrowPatch((dc * 0.3, dr * 0.3), (dc * 1.0, dr * 1.0),
                                     arrowstyle="-|>", mutation_scale=9 + 13 * w,
                                     color=BROWN, lw=0.7 + 3.2 * w, zorder=3))
        ax.text(dc * 1.42, dr * 1.42, f"{w:.0%}", ha="center", va="center",
                fontsize=8.5, color="#555")

    ax.scatter(0, 0, marker="s", s=200, color=BROWN, zorder=4)
    ax.set_title(title, fontsize=10, pad=8)
    ax.text(0, 2.0, sub, ha="center", fontsize=8.5, color="#333")

# the history that led in -- identical cell, drawn faintly
axes[1].add_patch(FancyArrowPatch((-1.9, -0.78), (-0.5, -0.78), arrowstyle="-|>",
                                  mutation_scale=9, color="#bbb", lw=1.4,
                                  linestyle="--", zorder=1))
axes[1].text(-1.2, -1.05, "arrived heading right", fontsize=7.5, color="#999",
             ha="center")

fig.tight_layout()
fig.savefig(OUT / "markov_memoryless.png", dpi=160, bbox_inches="tight")

# --- Figure 5: where clipping breaks it -----------------------------------
fig, ax = plt.subplots(figsize=(6.8, 2.2))
ax.set_xlim(-0.7, 7.7); ax.set_ylim(-0.95, 0.95)
ax.set_xticks(range(8)); ax.set_yticks([]); ax.set_aspect("equal")
ax.tick_params(labelsize=8)
for s in ("left", "right", "top"):
    ax.spines[s].set_visible(False)
ax.grid(True, axis="x", color=EDGE, lw=0.6)

# the "too close" cells are the ones NEXT TO the robot, not the robot's own
for c in (2, 4):
    ax.add_patch(Rectangle((c - 0.5, -0.45), 1, 0.9, facecolor="#f6d7d7", zorder=0))
ax.text(3, 0.62, "−3 zone", ha="center", fontsize=7.5, color="#c0392b")
ax.scatter(3, 0, marker="o", s=190, color=BLUE, zorder=4)
ax.text(3, -0.72, "robot", ha="center", fontsize=8, color=BLUE)

for col, label in [(5, "cow A"), (7, "cow B")]:
    ax.scatter(col, 0, marker="s", s=160, color=BROWN, zorder=4)
    ax.text(col, -0.72, label, ha="center", fontsize=8, color=BROWN)

ax.add_patch(FancyArrowPatch((4.9, 0.3), (4.1, 0.3), arrowstyle="-|>",
                             mutation_scale=10, color="#c0392b", lw=1.4))
ax.text(4.5, 0.80, "can be adjacent next step", ha="center", fontsize=7.5, color="#c0392b")
ax.add_patch(FancyArrowPatch((6.9, 0.3), (6.1, 0.3), arrowstyle="-|>",
                             mutation_scale=10, color="#2e8b57", lw=1.4))
ax.text(6.5, 0.80, "cannot", ha="center", fontsize=7.5, color="#2e8b57")

ax.set_title("Both cows encode to the same state, and their futures differ",
             fontsize=10, pad=8)
ax.set_xlabel("offsets +2 and +4; clipping sends both to +2", fontsize=8.5,
              color="#333", labelpad=4)
fig.tight_layout()
fig.savefig(OUT / "markov_clip_blur.png", dpi=160, bbox_inches="tight")
print("ok")
