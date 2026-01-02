"""Model evaluation and accuracy measurement for TVC controller."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import jax
import jax.numpy as jnp
import numpy as np
from flax import serialization

from .curriculum import CurriculumStage, build_curriculum
from .dynamics import RocketParams
from .env import TvcEnv
from .policies import PolicyConfig, PolicyFunctions, build_policy_network
from .training import RunningNormalizer

LOGGER = logging.getLogger(__name__)


@dataclass
class EpisodeMetrics:
    """Metrics for a single evaluation episode."""
    success: bool
    episode_return: float
    final_position_error: float
    final_velocity_error: float
    final_orientation_alignment: float
    mean_position_error: float
    mean_velocity_error: float
    episode_length: int
    crashed: bool
    out_of_bounds: bool


@dataclass
class EvaluationResults:
    """Comprehensive evaluation results."""
    success_rate: float
    mean_return: float
    std_return: float
    mean_position_error: float
    mean_velocity_error: float
    mean_orientation_alignment: float
    crash_rate: float
    out_of_bounds_rate: float
    mean_episode_length: float
    episodes: List[EpisodeMetrics] = field(default_factory=list)


def load_policy(
    policy_path: Path,
    policy_config: PolicyConfig = PolicyConfig(),
) -> tuple[Any, PolicyFunctions, RunningNormalizer | None]:
    """Load trained policy and optional normalizer from file."""
    policy_funcs = build_policy_network(policy_config)

    with open(policy_path, "rb") as f:
        data = f.read()

    # Try to decode as generic dict
    obs_rms = None
    try:
        loaded = serialization.from_bytes(None, data)
        
        # Check for new checkpoint format (contains "params", "opt_state", "obs_rms")
        if isinstance(loaded, dict) and "params" in loaded and "obs_rms" in loaded:
            params = loaded["params"]
            
            # Reconstruct RunningNormalizer from loaded dict
            # Flax deserializes dataclasses as dicts when target is None
            rms_dict = loaded["obs_rms"]
            if rms_dict:
                obs_rms = RunningNormalizer(
                    count=rms_dict.get("count", 0.0),
                    mean=jnp.array(rms_dict.get("mean", [])),
                    m2=jnp.array(rms_dict.get("m2", []))
                )
            
            LOGGER.info("✅ Loaded NEW policy format (includes ObsRMS)")
        else:
            # Legacy format or params-only dict
            params = loaded
            LOGGER.info("⚠️ Loaded LEGACY policy format (No ObsRMS)")

    except Exception as e:
        LOGGER.warning("Error loading policy: %s", e)
        raise

    LOGGER.info("Loaded policy from %s", policy_path)
    return params, policy_funcs, obs_rms


def evaluate_policy(
    params: Any,
    policy_funcs: PolicyFunctions,
    env: TvcEnv,
    stage: CurriculumStage,
    obs_rms: RunningNormalizer | None = None,
    num_episodes: int = 100,
    deterministic: bool = True,
    seed: int = 42,
) -> EvaluationResults:
    """Evaluate policy performance over multiple episodes.

    Args:
        params: Policy parameters
        policy_funcs: Policy functions
        env: Environment instance
        stage: Curriculum stage configuration
        obs_rms: Optional observation normalizer
        num_episodes: Number of episodes to evaluate
        deterministic: Use deterministic policy (no exploration noise)
        seed: Random seed for reproducibility

    Returns:
        Comprehensive evaluation results with accuracy metrics
    """
    LOGGER.info("=" * 80)
    LOGGER.info("🎯 EVALUATING POLICY: %s", stage.name)
    LOGGER.info("Episodes: %d | Deterministic: %s", num_episodes, deterministic)
    LOGGER.info("=" * 80)

    env.configure_stage(stage)
    rng = np.random.default_rng(seed)

    # Target values for error computation
    target_pos = np.array(stage.target_position, dtype=np.float32)
    target_vel = np.array(stage.target_velocity, dtype=np.float32)
    target_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

    episodes_data: List[EpisodeMetrics] = []

    for episode_idx in range(num_episodes):
        obs = env.reset()
        episode_return = 0.0
        done = False
        steps = 0

        position_errors = []
        velocity_errors = []
        orientation_alignments = []

        crashed = False
        out_of_bounds = False

        while not done and steps < env.max_steps:
            # Normalize observation if normalizer provided
            if obs_rms is not None:
                norm_obs = obs_rms.normalize(obs)
            else:
                norm_obs = jnp.asarray(obs, dtype=jnp.float32)

            # Get action from policy
            action = policy_funcs.actor(params, norm_obs, key=None, deterministic=deterministic)
            action_np = np.asarray(action, dtype=np.float32)

            # Step environment
            result = env.step(action_np)
            episode_return += result.reward
            obs = result.observation
            done = result.done
            steps += 1

            # Track errors during episode
            pos = env.data.qpos[0:3]
            vel = env.data.qvel[0:3]
            quat = env.data.qpos[3:7]

            pos_error = float(np.linalg.norm(pos - target_pos))
            vel_error = float(np.linalg.norm(vel - target_vel))
            orient_alignment = float(np.abs(np.dot(quat, target_quat)))

            position_errors.append(pos_error)
            velocity_errors.append(vel_error)
            orientation_alignments.append(orient_alignment)

            # Check failure modes
            if pos[2] < 0.2:
                crashed = True
            if np.linalg.norm(pos[0:2]) > 50.0 or pos[2] > 200.0:
                out_of_bounds = True

        # Final state evaluation
        final_pos_error = position_errors[-1] if position_errors else float('inf')
        final_vel_error = velocity_errors[-1] if velocity_errors else float('inf')
        final_orient_alignment = orientation_alignments[-1] if orientation_alignments else 0.0

        # Success criteria
        success = (
            final_pos_error < stage.position_tolerance and
            final_vel_error < stage.velocity_tolerance and
            final_orient_alignment > 0.95 and
            not crashed and
            not out_of_bounds
        )

        episode_metrics = EpisodeMetrics(
            success=success,
            episode_return=episode_return,
            final_position_error=final_pos_error,
            final_velocity_error=final_vel_error,
            final_orientation_alignment=final_orient_alignment,
            mean_position_error=float(np.mean(position_errors)),
            mean_velocity_error=float(np.mean(velocity_errors)),
            episode_length=steps,
            crashed=crashed,
            out_of_bounds=out_of_bounds,
        )
        episodes_data.append(episode_metrics)

        if (episode_idx + 1) % 10 == 0:
            current_success_rate = sum(e.success for e in episodes_data) / len(episodes_data)
            LOGGER.info("Episode %3d/%d | Success Rate: %.1f%% | Return: %7.2f | Pos Error: %.3fm",
                       episode_idx + 1, num_episodes, current_success_rate * 100,
                       episode_return, final_pos_error)

    # Aggregate statistics
    successes = [e.success for e in episodes_data]
    returns = [e.episode_return for e in episodes_data]
    pos_errors = [e.final_position_error for e in episodes_data]
    vel_errors = [e.final_velocity_error for e in episodes_data]
    orient_alignments = [e.final_orientation_alignment for e in episodes_data]
    crashes = [e.crashed for e in episodes_data]
    oob = [e.out_of_bounds for e in episodes_data]
    lengths = [e.episode_length for e in episodes_data]

    results = EvaluationResults(
        success_rate=float(np.mean(successes)),
        mean_return=float(np.mean(returns)),
        std_return=float(np.std(returns)),
        mean_position_error=float(np.mean(pos_errors)),
        mean_velocity_error=float(np.mean(vel_errors)),
        mean_orientation_alignment=float(np.mean(orient_alignments)),
        crash_rate=float(np.mean(crashes)),
        out_of_bounds_rate=float(np.mean(oob)),
        mean_episode_length=float(np.mean(lengths)),
        episodes=episodes_data,
    )

    # Print summary
    LOGGER.info("=" * 80)
    LOGGER.info("📊 EVALUATION RESULTS")
    LOGGER.info("=" * 80)
    LOGGER.info("Success Rate:       %.1f%% (%d/%d)", results.success_rate * 100,
                sum(successes), num_episodes)
    LOGGER.info("Mean Return:        %.2f ± %.2f", results.mean_return, results.std_return)
    LOGGER.info("Position Error:     %.3f m", results.mean_position_error)
    LOGGER.info("Velocity Error:     %.3f m/s", results.mean_velocity_error)
    LOGGER.info("Orientation:        %.3f (alignment)", results.mean_orientation_alignment)
    LOGGER.info("Crash Rate:         %.1f%%", results.crash_rate * 100)
    LOGGER.info("Out of Bounds:      %.1f%%", results.out_of_bounds_rate * 100)
    LOGGER.info("Mean Episode Len:   %.1f steps", results.mean_episode_length)
    LOGGER.info("=" * 80)

    return results


def evaluate_across_curriculum(
    policy_path: Path,
    output_dir: Path | None = None,
    num_episodes: int = 100,
    policy_config: PolicyConfig = PolicyConfig(),
    rocket_params: RocketParams = RocketParams(),
    seed: int = 42,
) -> Dict[str, EvaluationResults]:
    """Evaluate policy across all curriculum stages.

    Args:
        policy_path: Path to saved policy file
        output_dir: Optional directory to save results
        num_episodes: Episodes per stage
        policy_config: Policy configuration
        rocket_params: Rocket parameters
        seed: Random seed

    Returns:
        Dictionary mapping stage names to evaluation results
    """
    LOGGER.info("=" * 80)
    LOGGER.info("🚀 CURRICULUM EVALUATION")
    LOGGER.info("=" * 80)

    # Load policy
    params, policy_funcs, loaded_obs_rms = load_policy(policy_path, policy_config)

    # Create environment
    env = TvcEnv(dt=0.02, ctrl_limit=0.3, max_steps=2000, seed=seed)
    env.apply_rocket_params(rocket_params)

    # Initialize normalizer
    if loaded_obs_rms is not None:
        obs_rms = loaded_obs_rms
        LOGGER.info("✅ Using TRAINED observation normalizer from checkpoint")
    else:
        sample_obs = env.reset()
        obs_rms = RunningNormalizer.initialise(sample_obs)
        LOGGER.warning("⚠️ Using FRESH observation normalizer (Policy might fail due to input scaling mismatch)")

    # Build curriculum
    curriculum = build_curriculum()

    # Evaluate each stage
    stage_results: Dict[str, EvaluationResults] = {}

    for stage in curriculum:
        results = evaluate_policy(
            params=params,
            policy_funcs=policy_funcs,
            env=env,
            stage=stage,
            obs_rms=obs_rms,
            num_episodes=num_episodes,
            deterministic=True,
            seed=seed,
        )
        stage_results[stage.name] = results

    # Save results if output directory provided
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        results_path = output_dir / "evaluation_results.json"
        results_dict = {
            stage_name: {
                "success_rate": res.success_rate,
                "mean_return": res.mean_return,
                "std_return": res.std_return,
                "mean_position_error": res.mean_position_error,
                "mean_velocity_error": res.mean_velocity_error,
                "mean_orientation_alignment": res.mean_orientation_alignment,
                "crash_rate": res.crash_rate,
                "out_of_bounds_rate": res.out_of_bounds_rate,
                "mean_episode_length": res.mean_episode_length,
            }
            for stage_name, res in stage_results.items()
        }

        with open(results_path, "w") as f:
            json.dump(results_dict, f, indent=2)
        LOGGER.info("💾 Saved results: %s", results_path)

        # Generate comparison plots
        try:
            _generate_evaluation_plots(stage_results, output_dir)
            LOGGER.info("📊 Generated evaluation plots")
        except Exception as e:
            LOGGER.warning("Could not generate plots: %s", e)

    # Print overall summary
    LOGGER.info("=" * 80)
    LOGGER.info("📈 OVERALL CURRICULUM PERFORMANCE")
    LOGGER.info("=" * 80)
    for stage_name, results in stage_results.items():
        LOGGER.info("%-30s | Success: %5.1f%% | Return: %7.2f | Pos Error: %.3fm",
                   stage_name, results.success_rate * 100, results.mean_return,
                   results.mean_position_error)
    LOGGER.info("=" * 80)

    return stage_results


def _generate_evaluation_plots(stage_results: Dict[str, EvaluationResults], output_dir: Path) -> None:
    """Generate evaluation comparison plots."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')
    except ImportError:
        LOGGER.warning("Matplotlib not available, skipping plots")
        return

    stage_names = list(stage_results.keys())
    success_rates = [stage_results[name].success_rate * 100 for name in stage_names]
    mean_returns = [stage_results[name].mean_return for name in stage_names]
    pos_errors = [stage_results[name].mean_position_error for name in stage_names]
    vel_errors = [stage_results[name].mean_velocity_error for name in stage_names]
    crash_rates = [stage_results[name].crash_rate * 100 for name in stage_names]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Model Accuracy Evaluation Across Curriculum", fontsize=16, fontweight='bold')

    # Success rate
    axes[0, 0].bar(range(len(stage_names)), success_rates, color='green', alpha=0.7)
    axes[0, 0].set_xticks(range(len(stage_names)))
    axes[0, 0].set_xticklabels(stage_names, rotation=45, ha='right', fontsize=8)
    axes[0, 0].set_ylabel("Success Rate (%)")
    axes[0, 0].set_title("Success Rate by Stage")
    axes[0, 0].axhline(y=80, color='r', linestyle='--', alpha=0.5, label="80% target")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3, axis='y')

    # Mean return
    axes[0, 1].bar(range(len(stage_names)), mean_returns, color='blue', alpha=0.7)
    axes[0, 1].set_xticks(range(len(stage_names)))
    axes[0, 1].set_xticklabels(stage_names, rotation=45, ha='right', fontsize=8)
    axes[0, 1].set_ylabel("Mean Return")
    axes[0, 1].set_title("Mean Return by Stage")
    axes[0, 1].grid(True, alpha=0.3, axis='y')

    # Position error
    axes[0, 2].bar(range(len(stage_names)), pos_errors, color='orange', alpha=0.7)
    axes[0, 2].set_xticks(range(len(stage_names)))
    axes[0, 2].set_xticklabels(stage_names, rotation=45, ha='right', fontsize=8)
    axes[0, 2].set_ylabel("Position Error (m)")
    axes[0, 2].set_title("Mean Position Error by Stage")
    axes[0, 2].grid(True, alpha=0.3, axis='y')

    # Velocity error
    axes[1, 0].bar(range(len(stage_names)), vel_errors, color='purple', alpha=0.7)
    axes[1, 0].set_xticks(range(len(stage_names)))
    axes[1, 0].set_xticklabels(stage_names, rotation=45, ha='right', fontsize=8)
    axes[1, 0].set_ylabel("Velocity Error (m/s)")
    axes[1, 0].set_title("Mean Velocity Error by Stage")
    axes[1, 0].grid(True, alpha=0.3, axis='y')

    # Crash rate
    axes[1, 1].bar(range(len(stage_names)), crash_rates, color='red', alpha=0.7)
    axes[1, 1].set_xticks(range(len(stage_names)))
    axes[1, 1].set_xticklabels(stage_names, rotation=45, ha='right', fontsize=8)
    axes[1, 1].set_ylabel("Crash Rate (%)")
    axes[1, 1].set_title("Crash Rate by Stage")
    axes[1, 1].grid(True, alpha=0.3, axis='y')

    # Summary table
    axes[1, 2].axis('off')
    summary_text = "Overall Performance\n" + "=" * 25 + "\n\n"
    overall_success = np.mean(success_rates)
    overall_pos_error = np.mean(pos_errors)
    overall_vel_error = np.mean(vel_errors)
    overall_crash = np.mean(crash_rates)

    summary_text += f"Avg Success Rate: {overall_success:.1f}%\n"
    summary_text += f"Avg Pos Error:    {overall_pos_error:.3f}m\n"
    summary_text += f"Avg Vel Error:    {overall_vel_error:.3f}m/s\n"
    summary_text += f"Avg Crash Rate:   {overall_crash:.1f}%\n\n"
    summary_text += f"Stages Evaluated: {len(stage_names)}\n"

    axes[1, 2].text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
                   verticalalignment='center')

    plt.tight_layout()
    plot_path = output_dir / "evaluation_accuracy.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    LOGGER.info("📊 Plot saved: %s", plot_path)
