"""Training pipeline combining PPO, evolutionary strategies, and MPC."""
from __future__ import annotations

if __package__ in (None, ""):
    import pathlib
    import sys

    package_root = pathlib.Path(__file__).resolve().parents[1]
    if str(package_root) not in sys.path:
        sys.path.append(str(package_root))
    __package__ = "tvc"

import csv
import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax

from .curriculum import CurriculumStage, build_curriculum, select_stage
from .dynamics import RocketParams
from .env import StepResult, Tvc2DEnv
from .mpc import MpcConfig, compute_tvc_mpc_action
from .policies import PolicyConfig, build_policy_network, mutate_parameters


@dataclass(frozen=True)
class PpoEvolutionConfig:
    """Aggregates PPO and evolutionary hyperparameters."""

    gamma: float = 0.99
    lam: float = 0.95
    learning_rate: float = 8e-4
    clip_epsilon: float = 0.2
    rollout_length: int = 256
    num_epochs: int = 5
    minibatch_size: int = 128
    mutation_scale: float = 0.02
    population_size: int = 8
    elite_keep: int = 2
    policy_config: PolicyConfig = PolicyConfig()
    mpc_config: MpcConfig = MpcConfig(horizon=12, iterations=12, learning_rate=0.06)
    params: RocketParams = RocketParams()
    mpc_interval: int = 3
    evaluation_episodes: int = 1
    evaluation_max_steps: int = 250
    value_clip_epsilon: float = 0.2
    grad_clip_norm: float = 1.0
    use_lr_schedule: bool = True
    lr_warmup_fraction: float = 0.1
    min_learning_rate: float = 1e-5


@dataclass
class TrainingState:
    params: Any
    opt_state: optax.OptState
    elites: List[Any]
    rng: jax.Array
    obs_rms: "RunningNormalizer"
    history: List[Dict[str, Any]] = field(default_factory=list)
    update_step: int = 0
    lr_schedule: Callable[[int], float] | None = None


@dataclass
class RunningNormalizer:
    count: float = 0.0
    mean: jnp.ndarray = field(default_factory=lambda: jnp.zeros((3,), dtype=jnp.float32))
    m2: jnp.ndarray = field(default_factory=lambda: jnp.zeros((3,), dtype=jnp.float32))


def _build_optimizer(
    config: PpoEvolutionConfig,
    total_updates: int | None = None,
) -> Tuple[optax.GradientTransformation, Callable[[int], float] | None]:
    schedule: Callable[[int], float] | None = None
    learning_rate: Any = config.learning_rate

    if config.use_lr_schedule and total_updates and total_updates > 0:
        warmup_steps = max(1, int(config.lr_warmup_fraction * total_updates))
        decay_steps = max(1, total_updates - warmup_steps)
        base_schedule = optax.warmup_cosine_decay_schedule(
            init_value=config.min_learning_rate,
            peak_value=config.learning_rate,
            warmup_steps=warmup_steps,
            decay_steps=decay_steps,
            end_value=config.min_learning_rate,
        )
        learning_rate = base_schedule
        schedule = lambda step, _schedule=base_schedule: float(_schedule(step))

    transforms = []
    if config.grad_clip_norm and config.grad_clip_norm > 0.0:
        transforms.append(optax.clip_by_global_norm(config.grad_clip_norm))
    transforms.append(optax.adam(learning_rate))
    optimiser = optax.chain(*transforms)
    return optimiser, schedule


def _update_normalizer(rms: RunningNormalizer, observation: jnp.ndarray) -> RunningNormalizer:
    obs = jnp.asarray(observation, dtype=jnp.float32)
    count = rms.count + 1.0
    delta = obs - rms.mean
    mean = rms.mean + delta / count
    delta2 = obs - mean
    m2 = rms.m2 + delta * delta2
    return RunningNormalizer(count=count, mean=mean, m2=m2)


def _normalize_observation(rms: RunningNormalizer, observation: jnp.ndarray) -> jnp.ndarray:
    obs = jnp.asarray(observation, dtype=jnp.float32)
    if rms.count < 1.0:
        return obs
    variance = rms.m2 / rms.count
    variance = jnp.maximum(variance, 1e-6)
    return (obs - rms.mean) / jnp.sqrt(variance)


def _gather_state(env: Tvc2DEnv) -> jnp.ndarray:
    x = float(env.data.qpos[0])
    z = float(env.data.qpos[2])
    vx = float(env.data.qvel[0])
    vz = float(env.data.qvel[2])
    theta = env._pitch()
    omega = env._pitch_rate()
    return jnp.array([x, z, vx, vz, theta, omega], dtype=jnp.float32)


def _gaussian_log_prob(mean: jnp.ndarray, log_std: jnp.ndarray, action: jnp.ndarray) -> jnp.ndarray:
    var = jnp.exp(2.0 * log_std)
    return -0.5 * (jnp.sum(((action - mean) ** 2) / var + 2 * log_std + jnp.log(2 * jnp.pi)))


def _compute_gae(
    rewards: jnp.ndarray,
    values: jnp.ndarray,
    dones: jnp.ndarray,
    gamma: float,
    lam: float,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    def scan_fn(carry, inputs):
        reward, value, next_value, done = inputs
        delta = reward + gamma * (1.0 - done) * next_value - value
        advantage = delta + gamma * lam * (1.0 - done) * carry
        return advantage, advantage

    _, advantages_rev = jax.lax.scan(
        scan_fn,
        0.0,
        (rewards[::-1], values[:-1][::-1], values[1:][::-1], dones[::-1]),
    )
    advantages = advantages_rev[::-1]
    returns = advantages + values[:-1]
    return advantages, returns


def train_controller(
    env: Tvc2DEnv,
    total_episodes: int,
    rng: jax.Array,
    config: PpoEvolutionConfig = PpoEvolutionConfig(),
    output_dir: Path | None = None,
) -> TrainingState:
    """Runs PPO with evolutionary refinement and MPC guided actions."""

    policy_funcs = build_policy_network(config.policy_config)
    sample_obs = jnp.zeros((3,), dtype=jnp.float32)
    params = policy_funcs.init(rng, sample_obs)

    updates_per_epoch = max(1, math.ceil(config.rollout_length / config.minibatch_size))
    total_updates = total_episodes * config.num_epochs * updates_per_epoch
    optimiser, lr_schedule = _build_optimizer(config, total_updates)
    opt_state = optimiser.init(params)

    curriculum = build_curriculum()
    elites: List[Dict[str, object]] = []
    state = TrainingState(
        params=params,
        opt_state=opt_state,
        elites=elites,
        rng=rng,
        obs_rms=RunningNormalizer(),
        lr_schedule=lr_schedule,
    )
    logger = logging.getLogger("tvc.training")

    logger.info("Beginning training run for %s episodes", total_episodes)

    artifacts_dir: Path | None = None
    if output_dir is not None:
        artifacts_dir = Path(output_dir)
        artifacts_dir.mkdir(parents=True, exist_ok=True)

    for episode in range(total_episodes):
        stage = select_stage(curriculum, episode)
        trajectories, rollout_stats = _collect_rollout(env, stage, state, policy_funcs, config)
        state = _ppo_update(state, trajectories, optimiser, policy_funcs, config)
        episode_return = float(jnp.sum(trajectories["rewards"]))
        reward_mean = float(jnp.mean(trajectories["rewards"]))
        state, elite_best, elite_mean = _evolutionary_update(env, stage, state, policy_funcs, config)

        current_lr: float | None = None
        if state.lr_schedule is not None:
            current_lr = float(state.lr_schedule(state.update_step))

        episode_metrics: Dict[str, Any] = {
            "episode": float(episode + 1),
            "stage": stage.name,
            "disturbance_scale": float(stage.disturbance_scale),
            "rollout_return": episode_return,
            "reward_mean": reward_mean,
            "elite_best": elite_best,
            "elite_mean": elite_mean,
            "elites": float(len(state.elites)),
        }
        if current_lr is not None:
            episode_metrics["learning_rate"] = current_lr
        episode_metrics.update(rollout_stats)
        state.history.append(episode_metrics)

        logger.info(
            "Episode %s/%s | stage=%s (disturbance=%.2f) | rollout_return=%.3f | reward_mean=%.3f | elite_best=%.3f | elite_mean=%.3f | elites=%s",
            episode + 1,
            total_episodes,
            stage.name,
            stage.disturbance_scale,
            episode_return,
            reward_mean,
            elite_best,
            elite_mean,
            len(state.elites),
        )

    logger.info("Training loop complete. Elites maintained: %s", len(state.elites))
    if artifacts_dir is not None:
        save_training_artifacts(state.history, artifacts_dir, config)
    return state


def _collect_rollout(
    env: Tvc2DEnv,
    stage: CurriculumStage,
    state: TrainingState,
    funcs,
    config: PpoEvolutionConfig,
):
    obs_buffer: List[jnp.ndarray] = []
    action_buffer: List[jnp.ndarray] = []
    logprob_buffer: List[float] = []
    reward_buffer: List[float] = []
    value_buffer: List[float] = []
    done_buffer: List[float] = []

    observation = jnp.asarray(env.reset(disturbance_scale=stage.disturbance_scale), dtype=jnp.float32)
    obs_rms = state.obs_rms
    obs_rms = _update_normalizer(obs_rms, observation)
    norm_observation = _normalize_observation(obs_rms, observation)
    mpc_interval = max(1, int(config.mpc_interval))
    cached_mpc_action = jnp.zeros(2, dtype=jnp.float32)
    info_sums: Dict[str, float] | None = None
    info_abs_sums: Dict[str, float] | None = None
    info_max: Dict[str, float] | None = None
    info_min: Dict[str, float] | None = None

    for step in range(config.rollout_length):
        state.rng, policy_key = jax.random.split(state.rng)
        mean, log_std, value = funcs.distribution(state.params, norm_observation)
        std = jnp.exp(log_std)
        epsilon = jax.random.normal(policy_key, shape=mean.shape)
        policy_action = mean + std * epsilon
        if step % mpc_interval == 0:
            rocket_state = _gather_state(env)
            cached_mpc_action, _ = compute_tvc_mpc_action(
                rocket_state,
                stage.target_state,
                config.params,
                config.mpc_config,
            )
        combined_action = jnp.clip(
            policy_action + cached_mpc_action,
            -config.mpc_config.control_limit,
            config.mpc_config.control_limit,
        )

        log_prob = _gaussian_log_prob(mean, log_std, policy_action)

        result: StepResult = env.step(np.array(combined_action))

        obs_buffer.append(norm_observation)
        action_buffer.append(policy_action)
        logprob_buffer.append(float(log_prob))
        reward_buffer.append(result.reward)
        value_buffer.append(float(value))
        done_buffer.append(float(result.done))

        if info_sums is None:
            info_sums = {k: 0.0 for k in result.info}
            info_abs_sums = {k: 0.0 for k in result.info}
            info_max = {k: -float("inf") for k in result.info}
            info_min = {k: float("inf") for k in result.info}
        assert info_sums is not None
        assert info_abs_sums is not None
        assert info_max is not None
        assert info_min is not None

        for key, value_item in result.info.items():
            scalar = float(value_item)
            info_sums[key] += scalar
            info_abs_sums[key] += abs(scalar)
            info_max[key] = max(info_max[key], scalar)
            info_min[key] = min(info_min[key], scalar)

        observation = jnp.asarray(result.observation, dtype=jnp.float32)
        obs_rms = _update_normalizer(obs_rms, observation)
        norm_observation = _normalize_observation(obs_rms, observation)
        if result.done:
            observation = jnp.asarray(env.reset(disturbance_scale=stage.disturbance_scale), dtype=jnp.float32)
            obs_rms = _update_normalizer(obs_rms, observation)
            norm_observation = _normalize_observation(obs_rms, observation)

    # Bootstrap value for final state.
    value_buffer.append(float(funcs.value(state.params, norm_observation)))

    batch = {
        "observations": jnp.stack(obs_buffer),
        "actions": jnp.stack(action_buffer),
        "log_probs": jnp.array(logprob_buffer),
        "rewards": jnp.array(reward_buffer),
        "values": jnp.array(value_buffer),
        "dones": jnp.array(done_buffer),
    }
    state.obs_rms = obs_rms
    info_metrics: Dict[str, float] = {}
    num_steps = len(reward_buffer) or 1
    if info_sums is not None and info_abs_sums is not None and info_max is not None and info_min is not None:
        for key in info_sums:
            info_metrics[f"{key}_mean"] = info_sums[key] / num_steps
            info_metrics[f"{key}_abs_mean"] = info_abs_sums[key] / num_steps
            info_metrics[f"{key}_max"] = info_max[key]
            info_metrics[f"{key}_min"] = info_min[key]
    return batch, info_metrics


def _ppo_update(
    state: TrainingState,
    batch: Dict[str, jnp.ndarray],
    optimiser: optax.GradientTransformation,
    funcs,
    config: PpoEvolutionConfig,
) -> TrainingState:
    advantages, returns = _compute_gae(
        batch["rewards"],
        batch["values"],
        batch["dones"],
        config.gamma,
        config.lam,
    )

    advantages = (advantages - jnp.mean(advantages)) / (jnp.std(advantages) + 1e-8)

    dataset = {
        "obs": batch["observations"],
        "actions": batch["actions"],
        "log_probs": batch["log_probs"],
        "advantages": advantages,
        "returns": returns,
        "value_preds": batch["values"][:-1],
    }

    def loss_fn(params, minibatch):
        mean, log_std, values = funcs.distribution(params, minibatch["obs"])
        log_prob = jax.vmap(_gaussian_log_prob, in_axes=(0, None, 0))(mean, log_std, minibatch["actions"])
        ratio = jnp.exp(log_prob - minibatch["log_probs"])
        clipped = jnp.clip(ratio, 1.0 - config.clip_epsilon, 1.0 + config.clip_epsilon)
        actor_loss = -jnp.mean(jnp.minimum(ratio * minibatch["advantages"], clipped * minibatch["advantages"]))
        value_clipped = minibatch["value_preds"] + jnp.clip(
            values - minibatch["value_preds"],
            -config.value_clip_epsilon,
            config.value_clip_epsilon,
        )
        value_loss_unclipped = (minibatch["returns"] - values) ** 2
        value_loss_clipped = (minibatch["returns"] - value_clipped) ** 2
        value_loss = 0.5 * jnp.mean(jnp.maximum(value_loss_unclipped, value_loss_clipped))
        entropy = 0.5 * jnp.sum(1.0 + 2.0 * log_std + jnp.log(2.0 * jnp.pi))
        return actor_loss + value_loss - 0.001 * entropy

    params = state.params
    opt_state = state.opt_state
    num_samples = dataset["obs"].shape[0]

    for _ in range(config.num_epochs):
        permutation = np.random.permutation(num_samples)
        for start in range(0, num_samples, config.minibatch_size):
            idx = permutation[start : start + config.minibatch_size]
            minibatch = {k: v[idx] for k, v in dataset.items()}
            grads = jax.grad(loss_fn)(params, minibatch)
            updates, opt_state = optimiser.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            state.update_step += 1

    state.params = params
    state.opt_state = opt_state
    return state


def _evolutionary_update(
    env: Tvc2DEnv,
    stage: CurriculumStage,
    state: TrainingState,
    funcs,
    config: PpoEvolutionConfig,
) -> Tuple[TrainingState, float, float]:
    rng = state.rng
    population = [state.params] + state.elites
    while len(population) < config.population_size:
        rng, key = jax.random.split(rng)
        population.append(mutate_parameters(key, state.params, config.mutation_scale))

    scores = []
    for candidate in population:
        reward = _evaluate_candidate(env, stage, candidate, funcs, config, state.obs_rms)
        scores.append(reward)

    elite_indices = np.argsort(scores)[-config.elite_keep :][::-1]
    elites = [population[i] for i in elite_indices]
    state.elites = elites
    state.rng = rng

    best_reward = float(np.max(scores)) if scores else float("nan")
    mean_reward = float(np.mean(scores)) if scores else float("nan")
    return state, best_reward, mean_reward


def _evaluate_candidate(
    env: Tvc2DEnv,
    stage: CurriculumStage,
    params,
    funcs,
    config: PpoEvolutionConfig,
    normalizer: RunningNormalizer,
) -> float:
    total = 0.0
    base_normalizer = RunningNormalizer(
        count=float(normalizer.count),
        mean=jnp.array(normalizer.mean, copy=True),
        m2=jnp.array(normalizer.m2, copy=True),
    )
    for _ in range(config.evaluation_episodes):
        obs = jnp.asarray(env.reset(disturbance_scale=stage.disturbance_scale), dtype=jnp.float32)
        local_rms = _update_normalizer(base_normalizer, obs)
        norm_obs = _normalize_observation(local_rms, obs)
        done = False
        steps = 0
        while not done and steps < config.evaluation_max_steps:
            action = funcs.actor(params, norm_obs, None, True)
            result = env.step(np.array(action))
            total += result.reward
            obs = jnp.asarray(result.observation, dtype=jnp.float32)
            local_rms = _update_normalizer(local_rms, obs)
            norm_obs = _normalize_observation(local_rms, obs)
            done = result.done
            steps += 1
        base_normalizer = local_rms
    return total / max(1, config.evaluation_episodes)


def save_training_artifacts(
    history: List[Dict[str, Any]],
    output_dir: Path,
    config: PpoEvolutionConfig,
) -> None:
    """Persists per-episode metrics and diagnostic plots to ``output_dir``."""

    if not history:
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_csv = output_dir / "metrics.csv"
    metrics_json = output_dir / "metrics.json"
    field_names = sorted({key for entry in history for key in entry.keys()})

    with metrics_csv.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=field_names)
        writer.writeheader()
        for row in history:
            writer.writerow(row)

    with metrics_json.open("w", encoding="utf-8") as json_file:
        json.dump(history, json_file, indent=2)

    config_path = output_dir / "config.json"
    with config_path.open("w", encoding="utf-8") as config_file:
        json.dump(_config_to_serialisable(config), config_file, indent=2)

    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - optional dependency
        logging.getLogger("tvc.training").warning("matplotlib unavailable, skipping plots: %s", exc)
        return

    episodes = [entry.get("episode", idx + 1) for idx, entry in enumerate(history)]

    fig, axes = plt.subplots(2, 2, figsize=(13, 10), sharex="col")
    ax_rewards, ax_reward_mean = axes[0]
    ax_env, ax_lr = axes[1]

    def _plot_if_available(axis, key: str, label: str, entries: List[Dict[str, Any]]):
        values = [entry.get(key) for entry in entries]
        if any(value is None for value in values):
            return
        axis.plot(episodes, values, label=label)

    _plot_if_available(ax_rewards, "rollout_return", "Rollout return", history)
    _plot_if_available(ax_rewards, "elite_best", "Elite best", history)
    _plot_if_available(ax_rewards, "elite_mean", "Elite mean", history)
    ax_rewards.set_title("Episode returns")
    ax_rewards.set_ylabel("Reward")
    ax_rewards.grid(True, linestyle="--", alpha=0.3)
    ax_rewards.legend()

    _plot_if_available(ax_reward_mean, "reward_mean", "Reward mean", history)
    ax_reward_mean.set_title("Average reward per step")
    ax_reward_mean.set_ylabel("Reward")
    ax_reward_mean.grid(True, linestyle="--", alpha=0.3)

    env_metrics = [
        ("pitch_error_abs_mean", "|pitch error|"),
        ("pitch_rate_abs_mean", "|pitch rate|"),
        ("lateral_abs_mean", "|lateral|"),
        ("lateral_vel_abs_mean", "|lateral velocity|"),
    ]
    for key, label in env_metrics:
        _plot_if_available(ax_env, key, label, history)
    ax_env.set_title("Rocket stability diagnostics")
    ax_env.set_xlabel("Episode")
    ax_env.set_ylabel("Mean absolute value")
    ax_env.grid(True, linestyle="--", alpha=0.3)
    if ax_env.get_legend_handles_labels()[0]:
        ax_env.legend()

    if any("learning_rate" in entry for entry in history):
        _plot_if_available(ax_lr, "learning_rate", "Learning rate", history)
        ax_lr.set_ylabel("LR")
    else:
        ax_lr.text(0.5, 0.5, "Learning rate schedule disabled", ha="center", va="center")
    ax_lr.set_title("Optimizer schedule")
    ax_lr.set_xlabel("Episode")
    ax_lr.grid(True, linestyle="--", alpha=0.3)

    fig.tight_layout()
    plot_path = output_dir / "metrics.png"
    fig.savefig(str(plot_path), dpi=200)
    plt.close(fig)


def _config_to_serialisable(config: PpoEvolutionConfig) -> Dict[str, Any]:
    """Converts the configuration dataclass tree into JSON-friendly types."""

    activation_name = getattr(config.policy_config.activation, "__name__", str(config.policy_config.activation))

    return {
        "gamma": config.gamma,
        "lam": config.lam,
        "learning_rate": config.learning_rate,
        "clip_epsilon": config.clip_epsilon,
        "rollout_length": config.rollout_length,
        "num_epochs": config.num_epochs,
        "minibatch_size": config.minibatch_size,
        "mutation_scale": config.mutation_scale,
        "population_size": config.population_size,
        "elite_keep": config.elite_keep,
        "mpc_interval": config.mpc_interval,
        "evaluation_episodes": config.evaluation_episodes,
        "evaluation_max_steps": config.evaluation_max_steps,
        "value_clip_epsilon": config.value_clip_epsilon,
        "grad_clip_norm": config.grad_clip_norm,
        "use_lr_schedule": config.use_lr_schedule,
        "lr_warmup_fraction": config.lr_warmup_fraction,
        "min_learning_rate": config.min_learning_rate,
        "policy_config": {
            "hidden_dims": list(config.policy_config.hidden_dims),
            "activation": activation_name,
            "log_std_init": config.policy_config.log_std_init,
        },
        "mpc_config": {
            "horizon": config.mpc_config.horizon,
            "iterations": config.mpc_config.iterations,
            "learning_rate": config.mpc_config.learning_rate,
            "control_limit": config.mpc_config.control_limit,
        },
        "rocket_params": {
            "mass": config.params.mass,
            "inertia": config.params.inertia,
            "thrust": config.params.thrust,
            "arm": config.params.arm,
            "damping": config.params.damping,
            "gravity": config.params.gravity,
        },
    }
