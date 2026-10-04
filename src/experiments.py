"""
Every results table in the write-up, regenerated from the saved Q-tables.

The tables in README.md and docs/formulation.md came out of this, so anything
quoted there can be reproduced rather than taken on trust. Train first, then:

    python src/train.py --episodes 30000 --clip 2 --tag base
    python src/train.py --episodes 30000 --clip 3 --tag clip3
    python src/train.py --episodes 60000 --clip 3 --tag clip3_long
    python src/train.py --episodes 30000 --clip 2 --tag no_step --reward step=0
    python src/train.py --episodes 30000 --clip 2 --cow-bias 0.7 --tag chase2
    python src/train.py --episodes 30000 --clip 3 --cow-bias 0.7 --tag chase3
    python src/experiments.py

Everything is scored on the same 500 held-out seeds. The ablated policy is
scored against the *default* rewards: the ablation changes what the agent was
trained to want, not the yardstick it is measured with.
"""

from __future__ import annotations

import pathlib

import numpy as np

from cow_world import CowWorld
from evaluate import HEADER, eval_seeds, evaluate, time_near_the_cow
from policies import greedy_from_table, random_policy, rule_based

RESULTS = pathlib.Path(__file__).resolve().parent.parent / "results"


def load(tag: str, clip: int):
    path = RESULTS / f"q_{tag}.npy"
    if not path.exists():
        return None
    return greedy_from_table(np.load(path), clip)


def table(title: str, rows, env: CowWorld, seeds) -> None:
    print(f"\n{title}")
    print("=" * len(title))
    print(HEADER)
    present = [(n, p) for n, p in rows if p is not None]
    for name, policy in present:
        print(evaluate(name, policy, seeds, env).row())

    print("\n| policy             | steps in ring | closest |")
    print("|--------------------|---------------|---------|")
    for name, policy in present:
        frac, near = time_near_the_cow(policy, seeds, env)
        print(f"| {name:<18} | {frac:>12.2%} | {near:>7.2f} |")

    missing = [n for n, p in rows if p is None]
    if missing:
        print(f"\n  not trained yet: {', '.join(missing)}")


def main():
    seeds = eval_seeds(500)
    print(f"{len(seeds)} held-out seeds, {seeds[0]:,}..{seeds[-1]:,}. "
          f"Training uses seeds from 0.")

    table("Section 5 -- the original world",
          [("random", random_policy(np.random.default_rng(0))),
           ("rule-based", rule_based),
           ("Q, radius 2", load("base", 2)),
           ("Q, radius 3", load("clip3", 3)),
           ("Q, radius 3, 60k", load("clip3_long", 3))],
          CowWorld(), seeds)

    table("Section 7.1 -- removing the step penalty (scored on the default rewards)",
          [("rule-based", rule_based),
           ("Q, full reward", load("base", 2)),
           ("Q, no step penalty", load("no_step", 2))],
          CowWorld(), seeds)

    table("Section 7.2 -- the cow chases, 70% of the steps it moves",
          [("rule-based", rule_based),
           ("Q trained here, r2", load("chase2", 2)),
           ("Q trained here, r3", load("chase3", 3)),
           ("Q from easy world", load("base", 2))],
          CowWorld(cow_bias=0.7), seeds)


if __name__ == "__main__":
    main()
