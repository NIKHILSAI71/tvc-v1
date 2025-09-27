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
import datetime as _dt
import logging
import re
import sys
from dataclasses import replace
from pathlib import Path
from typing import Sequence

from .runtime import ensure_jax_runtime


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m tvc", description="TVC research CLI")
    subparsers = parser.add_subparsers(dest="command", metavar="command")

    train_parser = subparsers.add_parser(
        "train",
        help="Run the PPO + evolution training loop",
    )
    train_parser.add_argument(
    "--episodes",
    type=int,
    default=480,
    help="Number of training episodes to run",
    )
    train_parser.add_argument(
    "--seed",
    type=int,
    default=0,
    help="Seed for the JAX PRNG",
    )
    train_parser.add_argument(
    "--max-steps",
    type=int,
    default=400,
    help="Maximum environment steps before truncation",
    )
    train_parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"),
        type=str.upper,
        help="Logging verbosity for training output",
    )
    train_parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("training"),
        help="Directory where per-run artifacts will be stored",
    )
    train_parser.add_argument(
        "--run-tag",
        type=str,
        default=None,
        help="Optional suffix used when naming the run directory",
    )
    train_parser.add_argument(
        "--no-lr-schedule",
        action="store_true",
        help="Disable the warmup cosine learning-rate schedule",
    )
    train_parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Override the base learning rate used by PPO (default: 3e-4)",
    )
    train_parser.add_argument(
        "--entropy-coef",
        type=float,
        default=None,
        help="Override the entropy regularisation coefficient (default: 5e-3)",
    )
    train_parser.add_argument(
        "--reward-scale",
        type=float,
        default=None,
        help="Multiply raw rewards by this factor before optimisation (default: 1.0)",
    )
    train_parser.add_argument(
        "--rollout-length",
        type=int,
        default=None,
        help="Number of environment steps per PPO rollout (default: 512)",
    )
    train_parser.add_argument(
        "--policy-weight",
        type=float,
        default=None,
        help="Target policy contribution to the blended action (default: 0.8)",
    )
    train_parser.add_argument(
        "--mpc-weight",
        type=float,
        default=None,
        help="Target MPC contribution to the blended action (default: 0.2)",
    )
    train_parser.add_argument(
        "--policy-warmup-weight",
        type=float,
        default=None,
        help="Initial policy weight when progressive blending is enabled (default: 0.1)",
    )
    train_parser.add_argument(
        "--mpc-warmup-weight",
        type=float,
        default=None,
        help="Initial MPC weight when progressive blending is enabled (default: 0.9)",
    )
    train_parser.add_argument(
        "--blend-transition-episodes",
        type=int,
        default=None,
        help="Episodes per stage to transition from warmup to target blend weights (default: 120)",
    )
    train_parser.add_argument(
        "--no-progressive-blend",
        action="store_true",
        help="Keep MPC/policy weights fixed instead of interpolating over the stage",
    )
    train_parser.add_argument(
        "--plateau-global-warmup",
        type=int,
        default=None,
        help="Minimum global episodes before plateau LR scaling can trigger",
    )
    train_parser.add_argument(
        "--plateau-warmup",
        type=int,
        default=None,
        help="Stage-local warmup episodes before plateau LR scaling can trigger",
    )
    train_parser.add_argument(
        "--plateau-patience",
        type=int,
        default=None,
        help="Number of consecutive plateau episodes before LR is scaled",
    )
    train_parser.add_argument(
        "--plateau-threshold",
        type=float,
        default=None,
        help="Required improvement (in smoothed return) to reset plateau tracking",
    )
    train_parser.add_argument(
        "--plateau-factor",
        type=float,
        default=None,
        help="Multiplicative factor applied to LR when a plateau is detected",
    )
    train_parser.add_argument(
        "--enable-evolution",
        action="store_true",
        help="Enable mutation-based elite search after PPO updates",
    )
    train_parser.add_argument(
        "--evolution-population",
        type=int,
        default=None,
        help="Population size to use when evolution is enabled",
    )
    train_parser.add_argument(
        "--evolution-elite-keep",
        type=int,
        default=None,
        help="Number of elites retained each evolution round",
    )
    train_parser.add_argument(
        "--evolution-mutation-scale",
        type=float,
        default=None,
        help="Gaussian mutation scale applied during evolutionary rollout",
    )
    train_parser.add_argument(
        "--evolution-adoption-margin",
        type=float,
        default=None,
        help="Minimum reward improvement required before adopting an evolved policy",
    )
    train_parser.add_argument(
        "--policy-eval-interval",
        type=int,
        default=None,
        help="Episodes between policy-only evaluation sweeps (default: disabled)",
    )
    train_parser.add_argument(
        "--policy-eval-episodes",
        type=int,
        default=None,
        help="Number of episodes to average during evaluation runs",
    )
    train_parser.add_argument(
        "--policy-eval-max-steps",
        type=int,
        default=None,
        help="Maximum steps per evaluation episode",
    )
    train_parser.add_argument(
        "--disable-mpc-bc",
        action="store_true",
        help="Skip the MPC behaviour-cloning warmup before PPO",
    )
    train_parser.add_argument(
        "--mpc-bc-steps",
        type=int,
        default=None,
        help="Number of environment steps collected for MPC behaviour cloning",
    )
    train_parser.add_argument(
        "--mpc-bc-epochs",
        type=int,
        default=None,
        help="Number of optimisation epochs for MPC behaviour cloning",
    )
    train_parser.add_argument(
        "--mpc-bc-batch-size",
        type=int,
        default=None,
        help="Batch size used during MPC behaviour-cloning updates",
    )
    train_parser.add_argument(
        "--mpc-bc-learning-rate",
        type=float,
        default=None,
        help="Learning rate for the MPC behaviour-cloning optimiser",
    )
    train_parser.add_argument(
        "--mpc-bc-noise-scale",
        type=float,
        default=None,
        help="Standard deviation of exploratory noise applied to MPC actions during warmup",
    )
    train_parser.add_argument(
        "--mpc-bc-stage",
        type=str,
        default=None,
        help="Curriculum stage name used when collecting MPC behaviour-cloning data",
    )

    test_parser = subparsers.add_parser(
        "test",
        help="Execute package smoke tests via pytest",
    )
    test_parser.add_argument(
        "pytest_args",
        nargs=argparse.REMAINDER,
        help="Additional arguments forwarded to pytest (prefix with --)",
    )

    return parser.parse_args(argv)

def _configure_logging(level: str, log_path: Path | None = None) -> None:
    numeric = logging.getLevelName(level)
    if isinstance(numeric, str):  # unrecognised level name
        numeric = logging.INFO

    handlers = [logging.StreamHandler(sys.stdout)]
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path, mode="w", encoding="utf-8"))

    logging.basicConfig(
        level=numeric,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=handlers,
        force=True,
    )


def _sanitise_tag(tag: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]", "-", tag.strip())
    return cleaned or "run"


def _run_train(
    episodes: int,
    seed: int,
    max_steps: int,
    log_level: str,
    output_root: Path,
    run_tag: str | None,
    disable_schedule: bool,
    learning_rate: float | None,
    entropy_coef: float | None,
    reward_scale: float | None,
    rollout_length: int | None,
    policy_weight: float | None,
    mpc_weight: float | None,
    policy_warmup_weight: float | None,
    mpc_warmup_weight: float | None,
    blend_transition_episodes: int | None,
    disable_progressive_blend: bool,
    plateau_global_warmup: int | None,
    plateau_warmup: int | None,
    plateau_patience: int | None,
    plateau_threshold: float | None,
    plateau_factor: float | None,
    enable_evolution: bool,
    evolution_population: int | None,
    evolution_elite_keep: int | None,
    evolution_mutation_scale: float | None,
    evolution_adoption_margin: float | None,
    policy_eval_interval: int | None,
    policy_eval_episodes: int | None,
    policy_eval_max_steps: int | None,
    disable_mpc_bc: bool,
    mpc_bc_steps: int | None,
    mpc_bc_epochs: int | None,
    mpc_bc_batch_size: int | None,
    mpc_bc_learning_rate: float | None,
    mpc_bc_noise_scale: float | None,
    mpc_bc_stage: str | None,
) -> None:
    timestamp = _dt.datetime.now().strftime("run-%Y-%m-%d-%I-%M%p")
    run_name = _sanitise_tag(run_tag) if run_tag else timestamp
    run_dir = output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    log_path = run_dir / "training.log"
    _configure_logging(log_level, log_path)
    logger = logging.getLogger("tvc.cli")
    ensure_jax_runtime()

    import jax

    from .env import Tvc2DEnv
    from .training import PpoEvolutionConfig, train_controller

    key = jax.random.key(seed)
    env = Tvc2DEnv(max_steps=max_steps)
    config = PpoEvolutionConfig(use_lr_schedule=not disable_schedule)
    overrides = {}
    if plateau_global_warmup is not None:
        overrides["plateau_global_warmup_episodes"] = max(0, plateau_global_warmup)
    if plateau_warmup is not None:
        overrides["plateau_warmup_episodes"] = max(0, plateau_warmup)
    if plateau_patience is not None:
        overrides["plateau_patience"] = max(1, plateau_patience)
    if plateau_threshold is not None:
        overrides["plateau_threshold"] = float(plateau_threshold)
    if plateau_factor is not None:
        overrides["plateau_factor"] = float(plateau_factor)
    if learning_rate is not None:
        overrides["learning_rate"] = float(learning_rate)
    if entropy_coef is not None:
        overrides["entropy_coef"] = float(entropy_coef)
    if reward_scale is not None:
        overrides["reward_scale"] = float(reward_scale)
    if rollout_length is not None:
        overrides["rollout_length"] = max(1, int(rollout_length))
    if policy_weight is not None:
        overrides["policy_action_weight"] = float(policy_weight)
    if mpc_weight is not None:
        overrides["mpc_action_weight"] = float(mpc_weight)
    if policy_warmup_weight is not None:
        overrides["policy_action_weight_warmup"] = float(policy_warmup_weight)
    if mpc_warmup_weight is not None:
        overrides["mpc_action_weight_warmup"] = float(mpc_warmup_weight)
    if blend_transition_episodes is not None:
        overrides["action_blend_transition_episodes"] = max(1, int(blend_transition_episodes))
    if disable_progressive_blend:
        overrides["progressive_action_blend"] = False
    overrides["use_evolution"] = enable_evolution
    if evolution_population is not None:
        overrides["population_size"] = max(0, evolution_population)
    if evolution_elite_keep is not None:
        overrides["elite_keep"] = max(0, evolution_elite_keep)
    if evolution_mutation_scale is not None:
        overrides["mutation_scale"] = float(abs(evolution_mutation_scale))
    if evolution_adoption_margin is not None:
        overrides["evolution_adoption_margin"] = float(evolution_adoption_margin)
    if policy_eval_interval is not None:
        overrides["policy_eval_interval"] = max(0, policy_eval_interval)
    if policy_eval_episodes is not None:
        overrides["policy_eval_episodes"] = max(1, policy_eval_episodes)
    if policy_eval_max_steps is not None:
        overrides["policy_eval_max_steps"] = max(1, policy_eval_max_steps)
    if disable_mpc_bc:
        overrides["mpc_bc_enabled"] = False
    if mpc_bc_steps is not None:
        overrides["mpc_bc_steps"] = max(0, mpc_bc_steps)
    if mpc_bc_epochs is not None:
        overrides["mpc_bc_epochs"] = max(1, mpc_bc_epochs)
    if mpc_bc_batch_size is not None:
        overrides["mpc_bc_batch_size"] = max(1, mpc_bc_batch_size)
    if mpc_bc_learning_rate is not None:
        overrides["mpc_bc_learning_rate"] = float(mpc_bc_learning_rate)
    if mpc_bc_noise_scale is not None:
        overrides["mpc_bc_noise_scale"] = float(max(0.0, mpc_bc_noise_scale))
    if mpc_bc_stage is not None:
        overrides["mpc_bc_stage_name"] = str(mpc_bc_stage)
    if overrides:
        config = replace(config, **overrides)
    logger.info(
        (
            "Training started | episodes=%s seed=%s max_steps=%s lr=%.3g entropy_coef=%.3g "
            "reward_scale=%.3g rollout_length=%s progressive_blend=%s output_dir=%s"
        ),
        episodes,
        seed,
        max_steps,
        config.learning_rate,
        config.entropy_coef,
        config.reward_scale,
        config.rollout_length,
        config.progressive_action_blend,
        run_dir,
    )
    state = train_controller(env, total_episodes=episodes, rng=key, config=config, output_dir=run_dir)

    logger.info(
        "Training finished. episodes=%s elites_retained=%s artifacts=%s",
        episodes,
        len(state.elites),
        run_dir,
    )


def _run_tests(pytest_args: Sequence[str] | None) -> int:
    import pytest

    args = [] if not pytest_args else list(pytest_args)
    return pytest.main(args or ["-q"])


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv or sys.argv[1:])
    if args.command == "train":
        _run_train(
            episodes=args.episodes,
            seed=args.seed,
            max_steps=args.max_steps,
            log_level=args.log_level,
            output_root=args.output_root,
            run_tag=args.run_tag,
            disable_schedule=args.no_lr_schedule,
            learning_rate=args.learning_rate,
            entropy_coef=args.entropy_coef,
            reward_scale=args.reward_scale,
            rollout_length=args.rollout_length,
            policy_weight=args.policy_weight,
            mpc_weight=args.mpc_weight,
            policy_warmup_weight=args.policy_warmup_weight,
            mpc_warmup_weight=args.mpc_warmup_weight,
            blend_transition_episodes=args.blend_transition_episodes,
            disable_progressive_blend=args.no_progressive_blend,
            plateau_global_warmup=args.plateau_global_warmup,
            plateau_warmup=args.plateau_warmup,
            plateau_patience=args.plateau_patience,
            plateau_threshold=args.plateau_threshold,
            plateau_factor=args.plateau_factor,
            enable_evolution=args.enable_evolution,
            evolution_population=args.evolution_population,
            evolution_elite_keep=args.evolution_elite_keep,
            evolution_mutation_scale=args.evolution_mutation_scale,
            evolution_adoption_margin=args.evolution_adoption_margin,
            policy_eval_interval=args.policy_eval_interval,
            policy_eval_episodes=args.policy_eval_episodes,
            policy_eval_max_steps=args.policy_eval_max_steps,
            disable_mpc_bc=args.disable_mpc_bc,
            mpc_bc_steps=args.mpc_bc_steps,
            mpc_bc_epochs=args.mpc_bc_epochs,
            mpc_bc_batch_size=args.mpc_bc_batch_size,
            mpc_bc_learning_rate=args.mpc_bc_learning_rate,
            mpc_bc_noise_scale=args.mpc_bc_noise_scale,
            mpc_bc_stage=args.mpc_bc_stage,
        )
        return 0
    if args.command == "test":
        return _run_tests(args.pytest_args)

    print("No command specified. Use --help for usage information.")
    return 1


if __name__ == "__main__":  # pragma: no cover - exercised via CLI
    raise SystemExit(main())
