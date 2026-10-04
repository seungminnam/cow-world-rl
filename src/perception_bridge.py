"""
Optional challenge: drive the learned policy from a photograph instead of from
the grid.

A detector finds the cow in a real image, the box becomes a cell offset, and the
Q-table trained on the grid picks an action from it.

The interesting part is the join in the middle, because there is nothing
principled about it. The agent learned over cells. A camera gives a bearing and
an apparent size. Nothing in training says how many cells correspond to a box
filling 60% of the frame, so the thresholds below are mine, not the agent's, and
a different lens or a different distance would need different ones. That gap --
a policy that cannot accept observations from a sensor it was not trained
against without someone hand-writing the conversion -- is the same one that
keeps pretrained robot policies from dropping onto new hardware.

Needs ultralytics, which the rest of the project does not:

    pip install ultralytics
    python src/perception_bridge.py assets/cow.jpg
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from cow_world import ACTION_NAMES, GRID
from encoding import encode

COW_CLASS = "cow"
ROBOT_CELL = (4, 4)        # the robot is assumed mid-grid, so the cow can sit anywhere around it

# Fraction of the frame height the box has to fill to count as one cell away,
# two cells, and so on. Picked by eye from this photograph.
NEAR_THRESHOLDS = [(0.55, 1), (0.30, 2), (0.0, 3)]


def bearing_to_column_offset(cx: float, width: float, clip: int) -> int:
    """Where the cow sits across the frame becomes a sideways offset.

    Three bands rather than a continuous mapping, because the task asks for
    left, center or right and because a tabular agent cannot use more precision
    than its own cells carry.
    """
    frac = cx / width
    if frac < 1 / 3:
        return -clip
    if frac > 2 / 3:
        return clip
    return 0


def size_to_row_offset(box_h: float, height: float) -> int:
    """Apparent height stands in for distance. Bigger box, nearer cow."""
    frac = box_h / height
    for threshold, cells in NEAR_THRESHOLDS:
        if frac >= threshold:
            return cells
    return NEAR_THRESHOLDS[-1][1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--weights", default="yolo26n.pt")
    ap.add_argument("--q-table", default=None, help="defaults to results/q_base.npy")
    ap.add_argument("--clip", type=int, default=2)
    ap.add_argument("--out", default=None, help="write an annotated copy here")
    args = ap.parse_args()

    root = pathlib.Path(__file__).resolve().parent.parent
    q = np.load(args.q_table or root / "results" / "q_base.npy")

    from ultralytics import YOLO

    model = YOLO(args.weights)
    result = model(args.image, verbose=False)[0]

    cows = [b for b in result.boxes if model.names[int(b.cls)] == COW_CLASS]
    if not cows:
        print("no cow in the frame")
        return
    box = max(cows, key=lambda b: float(b.conf))

    h_img, w_img = result.orig_shape
    x1, y1, x2, y2 = box.xyxy[0].tolist()
    cx, box_h = (x1 + x2) / 2, y2 - y1

    dc = bearing_to_column_offset(cx, w_img, args.clip)
    dr = size_to_row_offset(box_h, h_img)
    dr = min(dr, args.clip)

    cow_cell = (min(max(ROBOT_CELL[0] + dr, 0), GRID - 1),
                min(max(ROBOT_CELL[1] + dc, 0), GRID - 1))
    obs = (*ROBOT_CELL, *cow_cell)
    state = encode(obs, args.clip)
    values = q[state]
    action = int(np.argmax(values))

    side = {-args.clip: "left", 0: "center", args.clip: "right"}[dc]
    print(f"detection      cow, confidence {float(box.conf):.2f}")
    print(f"box            ({x1:.0f}, {y1:.0f}) to ({x2:.0f}, {y2:.0f})"
          f"  -- {box_h / h_img:.0%} of frame height")
    print(f"read as        {side}, {dr} cell(s) away")
    print(f"agent state    robot {ROBOT_CELL}, cow {cow_cell}  ->  index {state}")
    print(f"action         {ACTION_NAMES[action]}")
    print()
    for name, value in zip(ACTION_NAMES, values):
        bar = "#" * max(0, int(round((value - values.min()) * 3)))
        print(f"  {name:<6} {value:7.2f}  {bar}")

    if args.out:
        import cv2

        img = cv2.imread(args.image)
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (219, 111, 31), 3)

        label = f"cow {float(box.conf):.2f} | {side}, {dr} cells | action: {ACTION_NAMES[action]}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.62, 2)
        # a box flush against the top of the frame leaves no room above it
        top = int(y1) - th - 12 if y1 - th - 12 > 0 else int(y1) + 4
        left = min(int(x1), img.shape[1] - tw - 14)
        cv2.rectangle(img, (left, top), (left + tw + 12, top + th + 12), (219, 111, 31), -1)
        cv2.putText(img, label, (left + 6, top + th + 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)
        cv2.imwrite(args.out, img)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
