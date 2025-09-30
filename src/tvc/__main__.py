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


def main() -> int:
    """Main CLI entry point."""
    import argparse

    from .dynamics import RocketParams
    from .mpc import MpcConfig
    from .policies import PolicyConfig
    from .training import TrainingConfig, train_controller

    parser = argparse.ArgumentParser(
        description="Train 3D TVC controller with PPO",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=1000,
        help="Total training episodes",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-4,
        help="PPO learning rate",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./tvc_output",
        help="Output directory for checkpoints and logs",
    )
    parser.add_argument(
        "--rollout-length",
        type=int,
        default=512,
        help="PPO rollout length",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=8,
        help="PPO epochs per update",
    )
    parser.add_argument(
        "--minibatch-size",
        type=int,
        default=64,
        help="PPO minibatch size",
    )

    args = parser.parse_args()

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
        LOGGER.info("=" * 60)

        return 0

    except KeyboardInterrupt:
        LOGGER.warning("Training interrupted by user")
        return 130
    except Exception as e:
        LOGGER.error("Training failed: %s", e, exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())