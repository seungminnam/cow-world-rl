"""
Tabular Q-learning.

The task offers two routes here, writing tabular Q-learning yourself or taking
DQN or PPO from Stable-Baselines3. I took the first. The state space is small
enough that the whole policy is a table of a few thousand numbers, and a deep RL
library would have put a wrapper between me and the thing the task is asking
about, which is what the agent learns and why.

SB3 still earns its place: tests/test_gym_api.py runs its env_checker against
the environment.

    Q(s,a) <- Q(s,a) + alpha * [ r + gamma * max_a' Q(s',a') - Q(s,a) ]

Sutton & Barto give this as equation 6.8 (sec 6.5, p.131).

    python src/train.py --episodes 30000 --clip 2 --tag base
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

from cow_world import MAX_STEPS, MOVES, CowWorld
from encoding import encode, n_states

RESULTS = pathlib.Path(__file__).resolve().parent.parent / "results"


def train(
    episodes: int = 30_000,
    clip: int = 2,
    alpha: float = 0.10,
    gamma: float = 0.95,
    eps_start: float = 1.0,
    eps_end: float = 0.05,
    eps_decay_frac: float = 0.6,
    seed: int = 0,
    env: CowWorld | None = None,
    rewards: dict | None = None,
    cow_bias: float = 0.0,
):
    """Returns (q_table, per-episode returns).

    Episode n trains on seed n, so training never touches the evaluation seeds,
    which start at 1,000,000.

    gamma is the one worth defending. The +20 lands roughly 16 steps out and
    0.95 ** 16 is about 0.44, so the goal still outweighs the step penalties
    piling up on the way. At 0.9 it would be 0.18 and stalling starts to look
    like a reasonable idea to the agent.
    """
    env = env or CowWorld(rewards=rewards, cow_bias=cow_bias)
    rng = np.random.default_rng(seed)      # seeded, so a training run replays
    q = np.zeros((n_states(clip), len(MOVES)), dtype=np.float64)
    returns = np.zeros(episodes)
    decay_over = max(1, int(episodes * eps_decay_frac))

    for ep in range(episodes):
        eps = max(eps_end, eps_start + (eps_end - eps_start) * (ep / decay_over))
        obs, _ = env.reset(seed=ep)
        state = encode(obs, clip)
        total = 0.0

        for _ in range(MAX_STEPS):
            if rng.random() < eps:
                action = int(rng.integers(len(MOVES)))
            else:
                action = int(np.argmax(q[state]))

            obs, reward, terminated, truncated, _ = env.step(action)
            nxt = encode(obs, clip)
            total += reward

            # Only `terminated` kills the bootstrap. A truncation at 60 steps
            # means the episode had a future and I cut it off, so zeroing it
            # there would teach the agent that late states are worthless.
            future = 0.0 if terminated else float(np.max(q[nxt]))
            q[state, action] += alpha * (reward + gamma * future - q[state, action])

            state = nxt
            if terminated or truncated:
                break

        returns[ep] = total

    return q, returns


def smooth(x: np.ndarray, window: int = 500) -> np.ndarray:
    if len(x) < window:
        return x
    return np.convolve(x, np.ones(window) / window, mode="valid")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=30_000)
    ap.add_argument("--clip", type=int, default=2)
    ap.add_argument("--tag", default="base")
    ap.add_argument("--cow-bias", type=float, default=0.0,
                    help="probability the cow heads for the robot instead of wandering")
    ap.add_argument("--reward", action="append", default=[],
                    metavar="TERM=VALUE",
                    help="override a reward term, e.g. --reward step=0")
    args = ap.parse_args()

    rewards = {}
    for item in args.reward:
        term, value = item.split("=")
        rewards[term] = float(value)

    RESULTS.mkdir(exist_ok=True)
    q, returns = train(episodes=args.episodes, clip=args.clip, rewards=rewards or None, cow_bias=args.cow_bias)

    np.save(RESULTS / f"q_{args.tag}.npy", q)
    np.save(RESULTS / f"returns_{args.tag}.npy", returns)

    meta = {
        "tag": args.tag,
        "clip": args.clip,
        "episodes": args.episodes,
        "rewards_overridden": rewards or None,
        "cow_bias": args.cow_bias,
        "states": n_states(args.clip),
        "states_visited": int((q != 0).any(axis=1).sum()),
        "mean_return_first_1000": round(float(returns[:1000].mean()), 3),
        "mean_return_last_1000": round(float(returns[-1000:].mean()), 3),
    }
    (RESULTS / f"meta_{args.tag}.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
