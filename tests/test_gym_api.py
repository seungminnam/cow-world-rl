"""
Checks the environment against Stable-Baselines3's env_checker.

The task lists SB3 under resources with the note that its check_env() validates
your environment, so this is their suggested check rather than mine. It is
worth more than my own tests on this particular question, because it was written
by people who did not know what I was going to build and it enforces the
Gymnasium contract rather than my reading of it.

SB3 is not needed for anything else here:

    pip install stable-baselines3
    python tests/test_gym_api.py
"""

from __future__ import annotations

import pathlib
import sys
import warnings

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from cow_world import CowWorld


def _check(env):
    from stable_baselines3.common.env_checker import check_env

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        check_env(env, warn=True, skip_render_check=True)
    return [str(w.message) for w in caught]


def test_the_default_environment_passes_check_env():
    assert _check(CowWorld()) == []


def test_the_harder_world_passes_too():
    """The chasing cow changes the dynamics, not the interface, but the point of
    a contract check is that I do not get to decide that."""
    assert _check(CowWorld(cow_bias=0.7)) == []


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"  ok  {name}")
    print("\nenvironment satisfies the Gymnasium contract")
