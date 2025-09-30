"""CLI for 3D TVC training."""

import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

LOGGER = logging.getLogger(__name__)


def cmd_train(args) -> int:
    """Train a new TVC controller model."""
    from .dynamics import RocketParams
    from .mpc import MpcConfig
    from .policies import PolicyConfig
    from .training import TrainingConfig, train_controller

    LOGGER.info("=" * 60)
    LOGGER.info("TVC 3D Training - Production Clean Version")
    LOGGER.info("=" * 60)
    LOGGER.info("Episodes: %d", args.episodes)
    LOGGER.info("Seed: %d", args.seed)
    LOGGER.info("Learning Rate: %.2e", args.learning_rate)
    LOGGER.info("Rollout Length: %d", args.rollout_length)
    LOGGER.info("Output: %s", args.output_dir)
    LOGGER.info("=" * 60)

    # Setup training configuration
    policy_config = PolicyConfig(
        hidden_dims=(512, 512, 256, 128),
        action_dims=3,
        action_limit=0.3,
        use_layer_norm=True,
    )

    mpc_config = MpcConfig(
        gimbal_limit=0.3,
        horizon=12,
        iterations=60,
    )

    rocket_params = RocketParams(
        mass=45.0,
        inertia=(120.0, 120.0, 8.0),
        thrust_max=600.0,
        thrust_min=200.0,
        gravity=9.81,
    )

    config = TrainingConfig(
        learning_rate=args.learning_rate,
        rollout_length=args.rollout_length,
        num_epochs=args.num_epochs,
        minibatch_size=args.minibatch_size,
        policy_config=policy_config,
        mpc_config=mpc_config,
        rocket_params=rocket_params,
    )

    # Run training
    try:
        final_state = train_controller(
            total_episodes=args.episodes,
            config=config,
            output_dir=Path(args.output_dir),
            seed=args.seed,
        )

        LOGGER.info("=" * 60)
        LOGGER.info("Training Complete!")
        LOGGER.info("Best Return: %.3f", final_state.best_return)
        LOGGER.info("Total Updates: %d", final_state.update_step)
        LOGGER.info("Success Rate: %.1f%%", (final_state.total_successes / args.episodes) * 100)
        LOGGER.info("=" * 60)

        return 0

    except KeyboardInterrupt:
        LOGGER.warning("Training interrupted by user")
        return 130
    except Exception as e:
        LOGGER.error("Training failed: %s", e, exc_info=True)
        return 1


def cmd_evaluate(args) -> int:
    """Evaluate a trained model's accuracy."""
    from .dynamics import RocketParams
    from .evaluate import evaluate_across_curriculum
    from .policies import PolicyConfig

    policy_path = Path(args.policy)
    output_dir = Path(args.output_dir) if args.output_dir else None

    LOGGER.info("=" * 60)
    LOGGER.info("TVC Model Evaluation")
    LOGGER.info("=" * 60)
    LOGGER.info("Policy: %s", policy_path)
    LOGGER.info("Episodes per stage: %d", args.episodes)
    LOGGER.info("=" * 60)

    if not policy_path.exists():
        LOGGER.error("Policy file not found: %s", policy_path)
        return 1

    try:
        stage_results = evaluate_across_curriculum(
            policy_path=policy_path,
            output_dir=output_dir,
            num_episodes=args.episodes,
            policy_config=PolicyConfig(),
            rocket_params=RocketParams(),
            seed=args.seed,
        )

        # Print summary
        LOGGER.info("")
        LOGGER.info("=" * 60)
        LOGGER.info("EVALUATION SUMMARY")
        LOGGER.info("=" * 60)

        for stage_name, results in stage_results.items():
            LOGGER.info("%-30s | Success: %5.1f%% | Pos Error: %.3fm",
                       stage_name, results.success_rate * 100, results.mean_position_error)

        overall_success = sum(r.success_rate for r in stage_results.values()) / len(stage_results)
        LOGGER.info("=" * 60)
        LOGGER.info("Overall Success Rate: %.1f%%", overall_success * 100)
        LOGGER.info("=" * 60)

        return 0

    except Exception as e:
        LOGGER.error("Evaluation failed: %s", e, exc_info=True)
        return 1


def cmd_test(args) -> int:
    """Run pytest test suite."""
    import subprocess

    LOGGER.info("=" * 60)
    LOGGER.info("Running TVC Test Suite (pytest)")
    LOGGER.info("=" * 60)

    # Build pytest command using python -m pytest (more reliable)
    pytest_args = [sys.executable, "-m", "pytest", "tests/"]

    if args.verbose:
        pytest_args.append("-v")

    if args.coverage:
        pytest_args.extend(["--cov=src/tvc", "--cov-report=term-missing"])

    if args.specific:
        # Replace tests/ with specific path
        pytest_args = [sys.executable, "-m", "pytest"] + args.specific.split()

    LOGGER.info("Command: pytest %s", " ".join(pytest_args[3:]))  # Show just pytest args
    LOGGER.info("=" * 60)

    try:
        result = subprocess.run(pytest_args, cwd=Path.cwd())
        return result.returncode

    except Exception as e:
        LOGGER.error("Test execution failed: %s", e, exc_info=True)
        LOGGER.error("Make sure pytest is installed: pip install pytest")
        return 1


def main() -> int:
    """Main CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="TVC 3D Rocket Controller - Training, Evaluation & Testing",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Train command
    train_parser = subparsers.add_parser(
        "train",
        help="Train a new TVC controller model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    train_parser.add_argument("--episodes", type=int, default=1000, help="Total training episodes (recommend 1000+ for full training)")
    train_parser.add_argument("--seed", type=int, default=42, help="Random seed")
    train_parser.add_argument("--learning-rate", type=float, default=5e-4, help="PPO learning rate (optimized)")
    train_parser.add_argument("--output-dir", type=str, default="./tvc_output", help="Output directory")
    train_parser.add_argument("--rollout-length", type=int, default=3072, help="PPO rollout length (optimized)")
    train_parser.add_argument("--num-epochs", type=int, default=4, help="PPO epochs per update")
    train_parser.add_argument("--minibatch-size", type=int, default=256, help="PPO minibatch size (optimized)")

    # Evaluate command
    eval_parser = subparsers.add_parser(
        "evaluate",
        help="Evaluate trained model accuracy across all curriculum stages",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    eval_parser.add_argument("--policy", type=str, default="tvc_output/policy_final.msgpack",
                            help="Path to trained policy file")
    eval_parser.add_argument("--episodes", type=int, default=100,
                            help="Episodes per curriculum stage")
    eval_parser.add_argument("--output-dir", type=str, default=None,
                            help="Output directory for evaluation results (optional)")
    eval_parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # Test command (pytest)
    test_parser = subparsers.add_parser(
        "test",
        help="Run pytest test suite",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    test_parser.add_argument("-v", "--verbose", action="store_true",
                            help="Verbose output")
    test_parser.add_argument("--coverage", action="store_true",
                            help="Run with coverage report")
    test_parser.add_argument("--specific", type=str, default=None,
                            help="Run specific test (e.g., 'tests/test_env_consistency.py::test_reset')")

    args = parser.parse_args()

    # Handle command routing
    if args.command == "train":
        return cmd_train(args)
    elif args.command == "evaluate":
        return cmd_evaluate(args)
    elif args.command == "test":
        return cmd_test(args)
    else:
        # Default behavior (backward compatibility) - run training
        # Add missing attributes for backward compatibility
        if not hasattr(args, 'episodes'):
            args.episodes = 500
        if not hasattr(args, 'seed'):
            args.seed = 42
        if not hasattr(args, 'learning_rate'):
            args.learning_rate = 3e-4
        if not hasattr(args, 'output_dir'):
            args.output_dir = "./tvc_output"
        if not hasattr(args, 'rollout_length'):
            args.rollout_length = 2048
        if not hasattr(args, 'num_epochs'):
            args.num_epochs = 4
        if not hasattr(args, 'minibatch_size'):
            args.minibatch_size = 128

        LOGGER.warning("No command specified. Use 'train', 'evaluate', or 'test'.")
        LOGGER.info("Example: python -m tvc train --episodes 500")
        LOGGER.info("         python -m tvc test --policy tvc_output/policy_final.msgpack")
        LOGGER.info("         python -m tvc evaluate --episodes 100")
        return 1


if __name__ == "__main__":
    sys.exit(main())