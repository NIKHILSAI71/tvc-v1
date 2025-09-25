"""Command-line interface for the :mod:`tvc` package.

This module enables convenient entry points so the package can be executed with
``python -m tvc <command>``. Two high-level commands are currently supported:

``train``
    Runs the training loop defined in :mod:`tvc.training` with a simplified
    configuration that targets quick experimentation. Hyperparameters can be
    tweaked via CLI flags.

``test``
    Executes the package's smoke tests using :mod:`pytest`. Any extra arguments
    after ``--`` are forwarded directly to ``pytest``.

The CLI is intentionally lightweight to keep the package importable in research
scripts while still offering an ergonomic way to kick the tyres or perform quick
checks from the terminal.
"""

from __future__ import annotations

import argparse
import sys
from typing import Sequence


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m tvc", description="TVC research CLI")
    subparsers = parser.add_subparsers(dest="command", metavar="command")

    train_parser = subparsers.add_parser("train", help="Run the PPO + evolution training loop")
    train_parser.add_argument("--episodes", type=int, default=10, help="Number of training episodes to run")
    train_parser.add_argument("--seed", type=int, default=0, help="Seed for the JAX PRNG")
    train_parser.add_argument(
        "--max-steps", type=int, default=400, help="Maximum environment steps before truncation"
    )

    test_parser = subparsers.add_parser("test", help="Execute package smoke tests via pytest")
    test_parser.add_argument(
        "pytest_args",
        nargs=argparse.REMAINDER,
        help="Additional arguments forwarded to pytest (prefix with --)",
    )

    return parser.parse_args(argv)


def _run_train(episodes: int, seed: int, max_steps: int) -> None:
    import jax

    from .env import Tvc2DEnv
    from .training import train_controller

    key = jax.random.key(seed)
    env = Tvc2DEnv(max_steps=max_steps)
    state = train_controller(env, total_episodes=episodes, rng=key)

    print("Training finished.")
    print(f"Episodes: {episodes}")
    print(f"Elite count: {len(state.elites)}")


def _run_tests(pytest_args: Sequence[str] | None) -> int:
    import pytest

    args = [] if not pytest_args else list(pytest_args)
    return pytest.main(args or ["-q"])


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv or sys.argv[1:])
    if args.command == "train":
        _run_train(args.episodes, args.seed, args.max_steps)
        return 0
    if args.command == "test":
        return _run_tests(args.pytest_args)

    print("No command specified. Use --help for usage information.")
    return 1


if __name__ == "__main__":  # pragma: no cover - exercised via CLI
    raise SystemExit(main())
