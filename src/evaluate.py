"""
The evaluation harness, written before the learning agent.

Every claim in this project is a comparison between two policies on a world that
moves by itself. That only means something if both policies face the same cow,
so the harness takes an explicit list of seeds and replays that same list for
everything it scores. No policy gets its own private episodes.

Run it directly to see the random and rule-based numbers:

    python src/evaluate.py
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cow_world import MAX_STEPS, CowWorld

# Training walks seeds up from 0, so evaluation starts far past wherever it can
# reach. Otherwise a good score could just mean the agent had seen those
# episodes before.
EVAL_SEED_START = 1_000_000


@dataclass
class Result:
    name: str
    episodes: int
    success_rate: float
    collision_rate: float
    timeout_rate: float
    steps_to_goal: float | None        # over successful episodes only
    mean_return: float

    def row(self) -> str:
        steps = "--" if self.steps_to_goal is None else f"{self.steps_to_goal:.1f}"
        return (f"| {self.name:<18} | {self.success_rate:>6.1%} | "
                f"{self.collision_rate:>8.1%} | {self.timeout_rate:>6.1%} | "
                f"{steps:>5} | {self.mean_return:>7.2f} |")


HEADER = ("| policy             | success | collision | timeout | steps | return  |\n"
          "|--------------------|---------|-----------|---------|-------|---------|")


def eval_seeds(n: int = 500, start: int = EVAL_SEED_START) -> list[int]:
    return list(range(start, start + n))


def run_episode(env: CowWorld, policy, seed: int) -> tuple[str, int, float]:
    """One episode. Returns (outcome, steps taken, total reward)."""
    obs, info = env.reset(seed=seed)
    total = 0.0
    for _ in range(MAX_STEPS):
        obs, reward, terminated, truncated, info = env.step(policy(obs, info))
        total += reward
        if terminated or truncated:
            return info.get("outcome") or "timeout", info["steps"], total
    return "timeout", MAX_STEPS, total


def evaluate(name: str, policy, seeds, env: CowWorld | None = None) -> Result:
    env = env or CowWorld()
    outcomes, goal_steps, returns = [], [], []

    for seed in seeds:
        outcome, steps, total = run_episode(env, policy, int(seed))
        outcomes.append(outcome)
        returns.append(total)
        if outcome == "goal":
            goal_steps.append(steps)

    n = len(outcomes)
    return Result(
        name=name,
        episodes=n,
        success_rate=outcomes.count("goal") / n,
        collision_rate=outcomes.count("collision") / n,
        timeout_rate=outcomes.count("timeout") / n,
        steps_to_goal=float(np.mean(goal_steps)) if goal_steps else None,
        mean_return=float(np.mean(returns)),
    )


def time_near_the_cow(policy, seeds, env: CowWorld | None = None) -> tuple[float, float]:
    """Share of steps spent inside the -3 ring, and the mean closest approach.

    Success rate alone cannot tell apart a policy that keeps its distance from
    one that skims past and gets lucky, and that difference is the whole
    safety question here.
    """
    env = env or CowWorld()
    close = total = 0
    closest = []

    for seed in seeds:
        obs, info = env.reset(seed=int(seed))
        nearest = 99
        for _ in range(MAX_STEPS):
            obs, _, terminated, truncated, info = env.step(policy(obs, info))
            from cow_world import chebyshev
            d = chebyshev(env.robot, env.cow)
            nearest = min(nearest, d)
            total += 1
            close += d == 1
            if terminated or truncated:
                break
        closest.append(nearest)

    return close / total, float(np.mean(closest))


if __name__ == "__main__":
    from policies import random_policy, rule_based

    seeds = eval_seeds()
    env = CowWorld()
    print(f"shared evaluation set: {len(seeds)} seeds, {seeds[0]}..{seeds[-1]}\n")
    print(HEADER)
    print(evaluate("random", random_policy(np.random.default_rng(0)), seeds, env).row())
    print(evaluate("rule-based", rule_based, seeds, env).row())
