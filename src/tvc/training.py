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
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
import optax
from flax import serialization

from .curriculum import CurriculumStage, build_curriculum
from .dynamics import RocketParams
from .env import StepResult, Tvc2DEnv
from .mpc import MpcConfig, compute_tvc_mpc_action
from .policies import PolicyConfig, build_policy_network, mutate_parameters


@dataclass(frozen=True)
class PpoEvolutionConfig:
    """Aggregates PPO and evolutionary hyperparameters."""

    gamma: float = 0.99
    lam: float = 0.95
    learning_rate: float = 3e-4
    clip_epsilon: float = 0.15
    rollout_length: int = 512
    num_epochs: int = 8
    minibatch_size: int = 64
    mutation_scale: float = 0.02
    population_size: int = 0
    elite_keep: int = 0
    use_evolution: bool = False
    evolution_adoption_margin: float = 10.0
    policy_config: PolicyConfig = PolicyConfig()
    mpc_config: MpcConfig = MpcConfig()
    params: RocketParams = RocketParams()
    mpc_interval: int = 3
    evaluation_episodes: int = 1
    evaluation_max_steps: int = 200
    policy_eval_interval: int = 0
    policy_eval_episodes: Optional[int] = None
    policy_eval_max_steps: Optional[int] = None
    value_clip_epsilon: float = 0.2
    grad_clip_norm: float = 0.5
    use_lr_schedule: bool = True
    lr_warmup_fraction: float = 0.2
    min_learning_rate: float = 5e-6
    lr_final_fraction: float = 0.4
    schedule_min_updates: int = 1500
    entropy_coef: float = 5e-3
    entropy_target: float = 0.6
    entropy_adjust_speed: float = 0.08
    entropy_coef_min: float = 1e-4
    entropy_coef_max: float = 2e-2
    value_coef: float = 1.0
    use_plateau_schedule: bool = True
    plateau_patience: int = 4
    plateau_factor: float = 0.6
    plateau_threshold: float = 1.0
    weight_decay: float = 1e-4
    plateau_global_warmup_episodes: int = 60
    adaptive_lr_enabled: bool = True
    adaptive_lr_target_kl: float = 0.015
    adaptive_lr_lower_ratio: float = 0.45
    adaptive_lr_upper_ratio: float = 1.8
    adaptive_lr_increase_factor: float = 1.2
    adaptive_lr_decrease_factor: float = 0.7
    adaptive_lr_min_scale: float = 0.05
    adaptive_lr_max_scale: float = 6.0
    stage_lr_bias: Dict[str, float] = field(
        default_factory=lambda: {
            "pad_hover": 1.0,
            "lateral_reject": 0.85,
            "wind_gust": 0.7,
        }
    )
    curriculum_adaptation: bool = True
    curriculum_reward_smoothing: float = 0.25
    reward_scale: float = 1.0
    policy_action_weight: float = 0.8
    mpc_action_weight: float = 0.2
    policy_action_weight_warmup: float = 0.1
    mpc_action_weight_warmup: float = 0.9
    action_blend_transition_episodes: int = 120
    progressive_action_blend: bool = True
    plateau_warmup_episodes: int = 40
    plateau_min_scale: float = 0.2
    mpc_loss_backoff_threshold: float = 2200.0
    mpc_loss_backoff_slope: float = 1.6
    mpc_loss_ema_decay: float = 0.45
    mpc_min_weight: float = 0.1
    mpc_backoff_warmup_episodes: int = 16
    mpc_backoff_reward_gate: float | None = None
    adaptive_lr_cooldown_episodes: int = 3
    mpc_bc_enabled: bool = False
    mpc_bc_steps: int = 0
    mpc_bc_batch_size: int = 256
    mpc_bc_epochs: int = 30
    mpc_bc_learning_rate: float = 1e-3
    mpc_bc_noise_scale: float = 0.015
    mpc_bc_stage_name: Optional[str] = "pad_hover"
    mpc_bc_log_every: int = 10
    mpc_bc_loss_reg: float = 1e-4


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
    lr_scale: float = 1.0
    lr_schedule_active: bool = False
    best_return: float = -float("inf")
    best_stage_reward: float = -float("inf")
    plateau_counter: int = 0
    last_improvement_episode: int = 0
    last_aux: Dict[str, float] = field(default_factory=dict)
    stage_index: int = 0
    stage_episode: int = 0
    stage_success_counter: int = 0
    stage_reward_ema: float = 0.0
    current_stage_name: Optional[str] = None
    last_mpc_plan: Optional[jnp.ndarray] = None
    mpc_loss_ema: float = float("nan")
    lr_adjust_cooldown: int = 0
    lr_adjust_last_direction: int = 0
    pretraining_metrics: Dict[str, Any] | None = None
    entropy_scale: float = 1.0


@dataclass
class RunningNormalizer:
    count: float = 0.0
    mean: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0,), dtype=jnp.float32))
    m2: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0,), dtype=jnp.float32))

    @classmethod
    def initialise(cls, observation: jnp.ndarray) -> "RunningNormalizer":
        obs = jnp.asarray(observation, dtype=jnp.float32)
        zeros = jnp.zeros_like(obs)
        return cls(count=0.0, mean=zeros, m2=zeros)


def _build_optimizer(
    config: PpoEvolutionConfig,
    total_updates: int | None = None,
) -> Tuple[optax.GradientTransformation, Callable[[int], float] | None]:
    schedule: Callable[[int], float] | None = None
    learning_rate: Any = config.learning_rate

    if config.use_lr_schedule and total_updates and total_updates >= max(1, config.schedule_min_updates):
        warmup_steps = max(1, int(config.lr_warmup_fraction * total_updates))
        decay_steps = max(1, total_updates - warmup_steps)
        end_value = max(config.min_learning_rate, config.learning_rate * config.lr_final_fraction)
        base_schedule = optax.warmup_cosine_decay_schedule(
            init_value=config.min_learning_rate,
            peak_value=config.learning_rate,
            warmup_steps=warmup_steps,
            decay_steps=decay_steps,
            end_value=end_value,
        )
        learning_rate = base_schedule

        def schedule_fn(step: int, _schedule=base_schedule) -> float:
            return float(_schedule(step))

        schedule = schedule_fn

    transforms = []
    if config.grad_clip_norm and config.grad_clip_norm > 0.0:
        transforms.append(optax.clip_by_global_norm(config.grad_clip_norm))
    transforms.append(optax.adamw(learning_rate=learning_rate, weight_decay=config.weight_decay))
    base_transform = optax.chain(*transforms)
    optimiser = optax.apply_if_finite(base_transform, max_consecutive_errors=5)
    return optimiser, schedule


def _update_normalizer(rms: RunningNormalizer, observation: jnp.ndarray) -> RunningNormalizer:
    obs = jnp.asarray(observation, dtype=jnp.float32)
    obs = jnp.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
    if rms.mean.size == 0 or rms.mean.shape != obs.shape or rms.count < 1.0:
        return RunningNormalizer(count=1.0, mean=obs, m2=jnp.zeros_like(obs))
    count = rms.count + 1.0
    delta = obs - rms.mean
    mean = rms.mean + delta / count
    delta2 = obs - mean
    m2 = rms.m2 + delta * delta2
    return RunningNormalizer(count=count, mean=mean, m2=m2)


def _normalize_observation(rms: RunningNormalizer, observation: jnp.ndarray) -> jnp.ndarray:
    obs = jnp.asarray(observation, dtype=jnp.float32)
    obs = jnp.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
    if rms.mean.size == 0 or rms.mean.shape != obs.shape or rms.count < 1.0:
        return obs
    variance = rms.m2 / rms.count
    variance = jnp.maximum(variance, 1e-6)
    return (obs - rms.mean) / jnp.sqrt(variance)


def _select_pretrain_stage(config: PpoEvolutionConfig, curriculum: List[CurriculumStage]) -> CurriculumStage | None:
    if not curriculum:
        return None
    if config.mpc_bc_stage_name is None:
        return curriculum[0]
    for stage in curriculum:
        if stage.name == config.mpc_bc_stage_name:
            return stage
    return curriculum[0]


def _mpc_behavior_cloning_warmup(
    env: Tvc2DEnv,
    stage: CurriculumStage | None,
    params: Any,
    funcs,
    config: PpoEvolutionConfig,
    rng: jax.Array,
    obs_rms: RunningNormalizer,
    logger: logging.Logger,
) -> Tuple[Any, RunningNormalizer, jax.Array, Dict[str, float]]:
    if not config.mpc_bc_enabled or config.mpc_bc_steps <= 0 or stage is None:
        return params, obs_rms, rng, {}

    steps = int(config.mpc_bc_steps)
    batch_size = max(1, int(config.mpc_bc_batch_size))
    epochs = max(1, int(config.mpc_bc_epochs))
    noise_scale = float(max(config.mpc_bc_noise_scale, 0.0))
    logger.info(
        "MPC behaviour-cloning warmup starting | stage=%s steps=%s epochs=%s batch=%s lr=%.3g noise=%.3g",
        stage.name,
        steps,
        epochs,
        batch_size,
        config.mpc_bc_learning_rate,
        noise_scale,
    )

    obs_list: List[jnp.ndarray] = []
    action_list: List[jnp.ndarray] = []
    warm_plan: jnp.ndarray | None = None
    if hasattr(env, "configure_stage"):
        env.configure_stage(stage)
    observation = jnp.asarray(env.reset(disturbance_scale=stage.disturbance_scale), dtype=jnp.float32)
    local_rms = obs_rms
    env_limit = float(getattr(env, "ctrl_limit", config.mpc_config.control_limit))
    action_limit = float(min(env_limit, config.mpc_config.control_limit))

    for step in range(steps):
        local_rms = _update_normalizer(local_rms, observation)
        norm_obs = _normalize_observation(local_rms, observation)
        rocket_state = _gather_state(env)
        mpc_action, _, warm_plan = compute_tvc_mpc_action(
            rocket_state,
            stage.target_state,
            config.params,
            config.mpc_config,
            warm_start=warm_plan,
        )
        if noise_scale > 0.0:
            rng, noise_key = jax.random.split(rng)
            noise = noise_scale * jax.random.normal(noise_key, shape=mpc_action.shape)
            mpc_action = jnp.clip(
                mpc_action + noise,
                -action_limit,
                action_limit,
            )
        obs_list.append(norm_obs)
        action_list.append(mpc_action)
        result = env.step(np.asarray(jnp.clip(mpc_action, -action_limit, action_limit), dtype=np.float32))
        observation = jnp.asarray(result.observation, dtype=jnp.float32)
        if result.done:
            if hasattr(env, "configure_stage"):
                env.configure_stage(stage)
            observation = jnp.asarray(env.reset(disturbance_scale=stage.disturbance_scale), dtype=jnp.float32)
            warm_plan = None

    dataset_obs = jnp.stack(obs_list)
    dataset_actions = jnp.stack(action_list)

    mean_initial, _, _ = funcs.distribution(params, dataset_obs, key=None, deterministic=True)
    initial_mse = jnp.mean(jnp.square(mean_initial - dataset_actions))

    optimiser = optax.adam(config.mpc_bc_learning_rate)
    opt_state = optimiser.init(params)

    def loss_fn(variables: Any, batch_obs: jnp.ndarray, batch_actions: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        mean, log_std, _ = funcs.distribution(variables, batch_obs, key=None, deterministic=True)
        mse = jnp.mean(jnp.square(mean - batch_actions))
        std_reg = jnp.mean(jnp.square(log_std - config.policy_config.log_std_init))
        loss = mse + float(config.mpc_bc_loss_reg) * std_reg
        return loss, mse

    @jax.jit
    def train_step(
        variables: Any,
        opt_state_in: optax.OptState,
        batch_obs: jnp.ndarray,
        batch_actions: jnp.ndarray,
    ) -> Tuple[Any, optax.OptState, jnp.ndarray, jnp.ndarray]:
        (loss_value, mse_value), grads = jax.value_and_grad(loss_fn, has_aux=True)(variables, batch_obs, batch_actions)
        updates, opt_state_out = optimiser.update(grads, opt_state_in, variables)
        variables = optax.apply_updates(variables, updates)
        return variables, opt_state_out, loss_value, mse_value

    num_samples = dataset_obs.shape[0]
    last_loss = jnp.array(0.0, dtype=jnp.float32)
    last_mse = jnp.array(0.0, dtype=jnp.float32)

    for epoch in range(epochs):
        perm = np.random.permutation(num_samples)
        for start in range(0, num_samples, batch_size):
            idx = perm[start : start + batch_size]
            if idx.size == 0:
                continue
            batch_obs = dataset_obs[idx]
            batch_actions = dataset_actions[idx]
            params, opt_state, last_loss, last_mse = train_step(params, opt_state, batch_obs, batch_actions)
        if config.mpc_bc_log_every > 0 and (epoch + 1) % config.mpc_bc_log_every == 0:
            logger.debug(
                "MPC BC epoch %s/%s | loss=%.5f mse=%.5f",
                epoch + 1,
                epochs,
                float(last_loss),
                float(last_mse),
            )

    mean_pred, _, _ = funcs.distribution(params, dataset_obs, key=None, deterministic=True)
    final_mse = jnp.mean(jnp.square(mean_pred - dataset_actions))
    metrics = {
        "bc_final_mse": float(final_mse),
        "bc_last_loss": float(last_loss),
        "bc_dataset_steps": float(num_samples),
        "bc_epochs": float(epochs),
        "bc_initial_mse": float(initial_mse),
    }

    logger.info(
        "MPC behaviour-cloning warmup complete | mse=%.5f -> %.5f loss=%.5f steps=%s",
        metrics["bc_initial_mse"],
        metrics["bc_final_mse"],
        metrics["bc_last_loss"],
        num_samples,
    )

    # Reset environment to a clean state before PPO rollouts.
    if hasattr(env, "configure_stage"):
        env.configure_stage(stage)
    env.reset(disturbance_scale=stage.disturbance_scale)
    return params, local_rms, rng, metrics


def _resolve_action_blend_weights(
    config: PpoEvolutionConfig,
    stage: CurriculumStage,
    stage_episode: int,
    *,
    progress_override: float | None = None,
) -> Tuple[float, float, float]:
    """Returns policy/MPC blending weights with optional scheduling."""

    if progress_override is not None:
        stage_progress = float(np.clip(progress_override, 0.0, 1.0))
    elif config.progressive_action_blend:
        transition_target = max(int(config.action_blend_transition_episodes), int(stage.min_episodes), 1)
        stage_progress = float(np.clip(stage_episode / transition_target, 0.0, 1.0))
    else:
        stage_progress = 1.0

    policy_target = max(float(config.policy_action_weight), 0.0)
    mpc_target = max(float(config.mpc_action_weight), 0.0)
    if config.progressive_action_blend:
        policy_start = max(float(config.policy_action_weight_warmup), 0.0)
        mpc_start = max(float(config.mpc_action_weight_warmup), 0.0)
        policy_weight_raw = (1.0 - stage_progress) * policy_start + stage_progress * policy_target
        mpc_weight_raw = (1.0 - stage_progress) * mpc_start + stage_progress * mpc_target
    else:
        policy_weight_raw = policy_target
        mpc_weight_raw = mpc_target

    weight_sum = policy_weight_raw + mpc_weight_raw
    if weight_sum <= 1e-6:
        return 0.5, 0.5, stage_progress

    policy_weight = policy_weight_raw / weight_sum
    mpc_weight = mpc_weight_raw / weight_sum
    return policy_weight, mpc_weight, stage_progress


def _apply_mpc_loss_backoff(
    policy_weight: float,
    mpc_weight: float,
    state: TrainingState,
    config: PpoEvolutionConfig,
) -> Tuple[float, float]:
    """Reduces the MPC contribution when optimisation loss remains high."""

    if mpc_weight <= 0.0:
        return policy_weight, mpc_weight
    threshold = float(config.mpc_loss_backoff_threshold)
    if not math.isfinite(threshold) or threshold <= 0.0:
        return policy_weight, mpc_weight
    warmup_limit = max(int(getattr(config, "mpc_backoff_warmup_episodes", 0)), 0)
    if getattr(state, "stage_episode", 0) < warmup_limit:
        return policy_weight, mpc_weight
    reward_gate = getattr(config, "mpc_backoff_reward_gate", None)
    if reward_gate is not None and math.isfinite(reward_gate):
        stage_reward = getattr(state, "stage_reward_ema", float("nan"))
        if not math.isfinite(stage_reward) or stage_reward < reward_gate:
            return policy_weight, mpc_weight
    loss_ema = getattr(state, "mpc_loss_ema", float("nan"))
    if not math.isfinite(loss_ema) or loss_ema <= threshold:
        return policy_weight, mpc_weight

    base_sum = max(policy_weight + mpc_weight, 1e-6)
    policy_share = policy_weight / base_sum
    mpc_share = mpc_weight / base_sum

    excess_ratio = max(0.0, (loss_ema - threshold) / threshold)
    slope = max(float(config.mpc_loss_backoff_slope), 0.0)
    scale = math.exp(-slope * excess_ratio)
    scaled_mpc_share = mpc_share * scale

    min_share = float(np.clip(config.mpc_min_weight, 0.0, 0.9))
    max_share = float(np.clip(1.0 - max(1e-3, min_share * 0.25), 0.1, 0.98))
    scaled_mpc_share = float(np.clip(scaled_mpc_share, min_share, max_share))
    scaled_policy_share = float(max(1e-6, 1.0 - scaled_mpc_share))

    total = scaled_policy_share + scaled_mpc_share
    if total <= 1e-6:
        return 0.5, 0.5
    return scaled_policy_share / total, scaled_mpc_share / total


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

    aligned_policy_config = replace(
        config.policy_config,
        action_limit=float(config.mpc_config.control_limit),
    )
    policy_funcs = build_policy_network(aligned_policy_config)
    curriculum = build_curriculum()
    initial_stage = curriculum[0] if curriculum else None
    logger = logging.getLogger("tvc.training")

    try:
        if hasattr(env, "apply_rocket_params"):
            env.apply_rocket_params(config.params)
    except Exception as exc:  # pragma: no cover - defensive path
        logger.warning("Unable to apply rocket parameters to environment: %s", exc)

    if initial_stage is not None and hasattr(env, "configure_stage"):
        env.configure_stage(initial_stage)
    disturbance_scale = float(initial_stage.disturbance_scale) if initial_stage else 1.0
    sample_obs = jnp.asarray(env.reset(disturbance_scale=disturbance_scale), dtype=jnp.float32)
    rng, init_key = jax.random.split(rng)
    params = policy_funcs.init(init_key, sample_obs)
    obs_rms = RunningNormalizer.initialise(sample_obs)
    pretrain_stage = _select_pretrain_stage(config, curriculum)
    params, obs_rms, rng, bc_metrics = _mpc_behavior_cloning_warmup(
        env,
        pretrain_stage,
        params,
        policy_funcs,
        config,
        rng,
        obs_rms,
        logger,
    )
    if initial_stage is not None and hasattr(env, "configure_stage"):
        env.configure_stage(initial_stage)
    sample_obs = jnp.asarray(env.reset(disturbance_scale=disturbance_scale), dtype=jnp.float32)
    obs_rms = _update_normalizer(obs_rms, sample_obs)

    updates_per_epoch = max(1, math.ceil(config.rollout_length / config.minibatch_size))
    total_updates = total_episodes * config.num_epochs * updates_per_epoch
    optimiser, lr_schedule = _build_optimizer(config, total_updates)
    opt_state = optimiser.init(params)

    eval_env: Optional[Tvc2DEnv] = None
    if config.policy_eval_interval and config.policy_eval_interval > 0:
        rng, eval_key = jax.random.split(rng)
        eval_seed = int(jax.random.randint(eval_key, shape=(), minval=0, maxval=2**31 - 1))
        eval_env = Tvc2DEnv(max_steps=env.max_steps, seed=eval_seed)
        try:
            if hasattr(eval_env, "apply_rocket_params"):
                eval_env.apply_rocket_params(config.params)
        except Exception as exc:  # pragma: no cover - defensive path
            logger.warning("Unable to apply rocket parameters to evaluation environment: %s", exc)

    elites: List[Dict[str, object]] = []
    state = TrainingState(
        params=params,
        opt_state=opt_state,
        elites=elites,
        rng=rng,
        obs_rms=obs_rms,
        lr_schedule=lr_schedule,
        current_stage_name=curriculum[0].name if curriculum else None,
        pretraining_metrics=bc_metrics if bc_metrics else None,
    )
    if config.adaptive_lr_enabled and curriculum:
        initial_bias = config.stage_lr_bias.get(curriculum[0].name)
        if initial_bias is not None:
            state.lr_scale = float(np.clip(initial_bias, config.adaptive_lr_min_scale, config.adaptive_lr_max_scale))

    logger.info("Beginning training run for %s episodes", total_episodes)

    artifacts_dir: Path | None = None
    if output_dir is not None:
        artifacts_dir = Path(output_dir)
        artifacts_dir.mkdir(parents=True, exist_ok=True)

    for episode in range(total_episodes):
        stage_idx = min(state.stage_index, len(curriculum) - 1)
        stage = curriculum[stage_idx]
        stage_name = stage.name
        stage_disturbance = float(stage.disturbance_scale)
        if hasattr(env, "configure_stage"):
            env.configure_stage(stage)
        trajectories, rollout_stats = _collect_rollout(env, stage, state, policy_funcs, config)
        state = _ppo_update(state, trajectories, optimiser, policy_funcs, config)
        _maybe_update_adaptive_lr(state, config)
        episode_return = float(jnp.sum(trajectories["rewards"]))
        reward_mean = float(jnp.mean(trajectories["rewards"]))
        state, elite_best, elite_mean, evolution_adopted = _evolutionary_update(
            env,
            stage,
            state,
            policy_funcs,
            config,
            optimiser,
            episode_return,
        )

        if state.lr_schedule is not None and state.lr_schedule_active:
            base_lr = state.lr_schedule(state.update_step)
        else:
            base_lr = config.learning_rate
        effective_lr = float(base_lr) * state.lr_scale
        stage, stage_metrics = _update_stage_progress(state, curriculum, stage, episode_return, logger, config)

        evaluation_metrics: Dict[str, Any] = {}
        if eval_env is not None and config.policy_eval_interval > 0 and (episode + 1) % config.policy_eval_interval == 0:
            eval_episodes = config.policy_eval_episodes or config.evaluation_episodes
            eval_max_steps = config.policy_eval_max_steps or config.evaluation_max_steps
            eval_return = _evaluate_candidate(
                eval_env,
                stage,
                state.params,
                policy_funcs,
                config,
                state.obs_rms,
                episodes=eval_episodes,
                max_steps=eval_max_steps,
            )
            evaluation_metrics["policy_eval_return"] = float(eval_return)

        episode_metrics: Dict[str, Any] = {
            "episode": float(episode + 1),
            "stage": stage_name,
            "disturbance_scale": stage_disturbance,
            "rollout_return": episode_return,
            "reward_mean": reward_mean,
            "elite_best": elite_best,
            "elite_mean": elite_mean,
            "elites": float(len(state.elites)),
            "learning_rate_base": float(base_lr),
            "learning_rate": effective_lr,
            "lr_scale": state.lr_scale,
            "lr_schedule_active": float(bool(state.lr_schedule_active)),
        }
        episode_metrics.update(rollout_stats)
        if state.last_aux:
            episode_metrics.update(state.last_aux)
        episode_metrics.update(stage_metrics)
        episode_metrics.update(evaluation_metrics)
        if config.use_evolution:
            episode_metrics["evolution_adopted"] = float(bool(evolution_adopted))

        global_episode = episode + 1
        state.best_return = max(state.best_return, episode_return)
        episodes_since_improvement_metric = global_episode - max(state.last_improvement_episode, 0)

        if config.use_plateau_schedule:
            stage_episode = float(episode_metrics.get("stage_episode", 0.0))
            smoothed_return = float(state.stage_reward_ema)
            if math.isfinite(smoothed_return) and smoothed_return > state.best_stage_reward + config.plateau_threshold:
                state.best_stage_reward = smoothed_return
                state.plateau_counter = 0
                state.last_improvement_episode = global_episode
            else:
                warmup_passed = (
                    stage_episode >= float(config.plateau_warmup_episodes)
                    and global_episode >= int(config.plateau_global_warmup_episodes)
                )
                if not warmup_passed:
                    state.plateau_counter = 0
                else:
                    state.plateau_counter += 1
                    episodes_since_improvement = global_episode - max(state.last_improvement_episode, 0)
                    if (
                        state.plateau_counter >= config.plateau_patience
                        and episodes_since_improvement >= config.plateau_patience
                    ):
                        min_floor = max(
                            config.min_learning_rate / max(config.learning_rate, 1e-12),
                            config.plateau_min_scale,
                        )
                        scaled_scale = max(state.lr_scale * config.plateau_factor, min_floor)
                        if scaled_scale < state.lr_scale - 1e-9:
                            state.lr_scale = scaled_scale
                            logger.info(
                                "Learning-rate plateau detected (episode %s, stage=%s, ema=%.3f, best_ema=%.3f). Scaling LR to %.6e",
                                global_episode,
                                stage_name,
                                smoothed_return,
                                state.best_stage_reward,
                                float(base_lr) * state.lr_scale,
                            )
                        state.plateau_counter = 0
            episodes_since_improvement_metric = global_episode - max(state.last_improvement_episode, 0)
        if config.adaptive_lr_enabled:
            lower_bound = max(config.adaptive_lr_min_scale, config.plateau_min_scale)
            state.lr_scale = float(np.clip(state.lr_scale, lower_bound, config.adaptive_lr_max_scale))
        best_stage_metric = state.best_stage_reward if math.isfinite(state.best_stage_reward) else float("nan")
        episode_metrics["plateau_counter"] = float(state.plateau_counter)
        episode_metrics["best_return"] = state.best_return
        episode_metrics["best_stage_reward"] = best_stage_metric
        episode_metrics["episodes_since_improvement"] = float(episodes_since_improvement_metric)
        state.history.append(episode_metrics)

        log_stage_progress = float(episode_metrics.get("action_blend_progress", float("nan")))
        raw_reward_mean = float(episode_metrics.get("reward_raw_mean", float("nan")))
        entropy_metric = float(episode_metrics.get("entropy", float("nan")))
        mpc_loss_metric = float(episode_metrics.get("mpc_loss_mean", float("nan")))
        logger.info(
            (
                "Episode %s/%s | stage=%s (disturbance=%.2f, progress=%.2f) | "
                "return=%.3f | reward_mean=%.3f (raw=%.3f) | elite_best=%.3f | "
                "lr=%.3e | entropy=%.3f | mpc_loss=%.3f | elites=%s"
            ),
            episode + 1,
            total_episodes,
            stage_name,
            stage_disturbance,
            log_stage_progress,
            episode_return,
            reward_mean,
            raw_reward_mean,
            elite_best,
            effective_lr,
            entropy_metric,
            mpc_loss_metric,
            len(state.elites),
        )

    logger.info("Training loop complete. Elites maintained: %s", len(state.elites))
    if artifacts_dir is not None:
        save_training_artifacts(state.history, artifacts_dir, config, state.pretraining_metrics)
        checkpoint_paths = save_policy_checkpoints(state, artifacts_dir)
        logger.info(
            "Saved policy checkpoints: %s",
            ", ".join(path.name for path in checkpoint_paths.values()),
        )
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
    applied_action_buffer: List[jnp.ndarray] = []
    mpc_action_buffer: List[jnp.ndarray] = []
    logprob_buffer: List[float] = []
    reward_buffer: List[float] = []
    raw_reward_buffer: List[float] = []
    value_buffer: List[float] = []
    done_buffer: List[float] = []

    if hasattr(env, "configure_stage"):
        env.configure_stage(stage)
    observation = jnp.asarray(env.reset(disturbance_scale=stage.disturbance_scale), dtype=jnp.float32)
    obs_rms = state.obs_rms
    obs_rms = _update_normalizer(obs_rms, observation)
    norm_observation = _normalize_observation(obs_rms, observation)
    mpc_interval = max(1, int(config.mpc_interval))
    cached_mpc_action = jnp.zeros(2, dtype=jnp.float32)
    cached_plan = state.last_mpc_plan
    if cached_plan is not None:
        cached_plan = jnp.asarray(cached_plan, dtype=jnp.float32)
    env_limit = float(getattr(env, "ctrl_limit", config.mpc_config.control_limit))
    action_limit = float(min(env_limit, config.mpc_config.control_limit))
    reward_scale = float(config.reward_scale)
    reward_scale = reward_scale if reward_scale > 0.0 else 1.0
    policy_weight_raw, mpc_weight_raw, stage_progress = _resolve_action_blend_weights(
        config,
        stage,
        state.stage_episode,
    )
    policy_weight, mpc_weight = _apply_mpc_loss_backoff(policy_weight_raw, mpc_weight_raw, state, config)
    info_sums: Dict[str, float] | None = None
    info_abs_sums: Dict[str, float] | None = None
    info_max: Dict[str, float] | None = None
    info_min: Dict[str, float] | None = None
    mpc_loss_values: List[float] = []
    mpc_grad_norms: List[float] = []
    mpc_iterations: List[float] = []
    mpc_saturation: List[float] = []
    normalised_reward_buffer: List[float] = []
    policy_action_resets = 0
    mpc_action_resets = 0
    blended_action_resets = 0
    std_min_bound = float(math.exp(config.policy_config.log_std_min))
    std_max_bound = float(math.exp(config.policy_config.log_std_max))
    std_fallback = float(math.exp(config.policy_config.log_std_init))

    for step in range(config.rollout_length):
        state.rng, policy_key, dropout_key = jax.random.split(state.rng, 3)
        use_dropout = config.policy_config.dropout_rate > 0.0
        mean, log_std, value = funcs.distribution(
            state.params,
            norm_observation,
            key=dropout_key if use_dropout else None,
            deterministic=not use_dropout,
        )
        mean = jnp.nan_to_num(mean, nan=0.0, posinf=action_limit, neginf=-action_limit)
        log_std = jnp.nan_to_num(
            log_std,
            nan=config.policy_config.log_std_init,
            posinf=config.policy_config.log_std_max,
            neginf=config.policy_config.log_std_min,
        )
        log_std = jnp.clip(log_std, config.policy_config.log_std_min, config.policy_config.log_std_max)
        value = jnp.nan_to_num(value, nan=0.0)
        std = jnp.exp(log_std)
        std = jnp.nan_to_num(std, nan=std_fallback)
        std = jnp.clip(std, std_min_bound, std_max_bound)
        epsilon = jax.random.normal(policy_key, shape=mean.shape)
        policy_action = mean + std * epsilon
        policy_action = jnp.nan_to_num(policy_action, nan=0.0, posinf=action_limit, neginf=-action_limit)
        if not bool(jnp.all(jnp.isfinite(policy_action))):
            policy_action_resets += 1
            policy_action = mean
        policy_action = jnp.clip(policy_action, -action_limit, action_limit)
        if step % mpc_interval == 0:
            rocket_state = _gather_state(env)
            cached_mpc_action, mpc_diag, cached_plan = compute_tvc_mpc_action(
                rocket_state,
                stage.target_state,
                config.params,
                config.mpc_config,
                warm_start=cached_plan,
            )
            mpc_loss_value = float(mpc_diag.get("mpc_loss", 0.0))
            if not math.isfinite(mpc_loss_value):
                mpc_loss_value = 0.0
            mpc_loss_values.append(mpc_loss_value)
            mpc_grad = float(mpc_diag.get("mpc_grad_norm", 0.0))
            if not math.isfinite(mpc_grad):
                mpc_grad = 0.0
            mpc_grad_norms.append(mpc_grad)
            mpc_iters_val = float(mpc_diag.get("mpc_iterations", config.mpc_config.iterations))
            if not math.isfinite(mpc_iters_val):
                mpc_iters_val = float(config.mpc_config.iterations)
            mpc_iterations.append(mpc_iters_val)
            mpc_sat_val = float(mpc_diag.get("mpc_saturation", 0.0))
            if not math.isfinite(mpc_sat_val):
                mpc_sat_val = 0.0
            mpc_saturation.append(mpc_sat_val)
            cached_mpc_action = jnp.nan_to_num(cached_mpc_action, nan=0.0, posinf=action_limit, neginf=-action_limit)
            if not bool(jnp.all(jnp.isfinite(cached_mpc_action))):
                mpc_action_resets += 1
                cached_plan = None
                cached_mpc_action = jnp.zeros_like(cached_mpc_action)
            cached_mpc_action = jnp.clip(cached_mpc_action, -action_limit, action_limit)
        else:
            cached_mpc_action = jnp.clip(jnp.nan_to_num(cached_mpc_action, nan=0.0, posinf=action_limit, neginf=-action_limit), -action_limit, action_limit)
        blended_action = policy_weight * policy_action + mpc_weight * cached_mpc_action
        combined_action = jnp.clip(blended_action, -action_limit, action_limit)
        combined_action = jnp.nan_to_num(combined_action, nan=0.0, posinf=action_limit, neginf=-action_limit)

        log_prob = _gaussian_log_prob(mean, log_std, policy_action)
        log_prob = jnp.nan_to_num(log_prob, nan=-1e6, posinf=0.0, neginf=-1e6)

        combined_action_np = np.asarray(combined_action, dtype=np.float32)
        if not np.isfinite(combined_action_np).all():
            blended_action_resets += 1
            fallback_np = np.asarray(np.clip(cached_mpc_action, -action_limit, action_limit), dtype=np.float32)
            combined_action_np = fallback_np
            combined_action = jnp.asarray(combined_action_np)
        combined_action = jnp.nan_to_num(combined_action, nan=0.0, posinf=action_limit, neginf=-action_limit)
        result: StepResult = env.step(combined_action_np)
        result_info = result.info
        normalised_reward = float(result_info.get("reward_normalised", result.reward))
        raw_reward = float(result_info.get("raw_reward", normalised_reward))
        scaled_reward = raw_reward * reward_scale
        normalised_reward_buffer.append(normalised_reward)

        obs_buffer.append(norm_observation)
        action_buffer.append(jnp.nan_to_num(policy_action, nan=0.0, posinf=action_limit, neginf=-action_limit))
        applied_action_buffer.append(jnp.nan_to_num(combined_action, nan=0.0, posinf=action_limit, neginf=-action_limit))
        mpc_action_buffer.append(jnp.nan_to_num(cached_mpc_action, nan=0.0, posinf=action_limit, neginf=-action_limit))
        logprob_buffer.append(float(log_prob))
        reward_buffer.append(scaled_reward)
        raw_reward_buffer.append(raw_reward)
        value_buffer.append(float(jnp.nan_to_num(value, nan=0.0)))
        done_buffer.append(float(result.done))

        if info_sums is None:
            info_sums = {k: 0.0 for k in result.info}
            info_abs_sums = {k: 0.0 for k in result.info}
            info_max = {k: -float("inf") for k in result.info}
            info_min = {k: float("inf") for k in result.info}

        assert info_sums is not None and info_abs_sums is not None and info_max is not None and info_min is not None

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
            if hasattr(env, "configure_stage"):
                env.configure_stage(stage)
            observation = jnp.asarray(env.reset(disturbance_scale=stage.disturbance_scale), dtype=jnp.float32)
            obs_rms = _update_normalizer(obs_rms, observation)
            norm_observation = _normalize_observation(obs_rms, observation)

    # Bootstrap value for final state.
    bootstrap_value = funcs.value(state.params, norm_observation)
    value_buffer.append(float(jnp.nan_to_num(bootstrap_value, nan=0.0)))

    observations = jnp.nan_to_num(jnp.stack(obs_buffer), nan=0.0)
    actions = jnp.nan_to_num(
        jnp.stack(action_buffer),
        nan=0.0,
        posinf=action_limit,
        neginf=-action_limit,
    )
    log_probs = jnp.nan_to_num(jnp.array(logprob_buffer), nan=-1e6, posinf=0.0, neginf=-1e6)
    log_probs = jnp.clip(log_probs, -1e6, 0.0)
    rewards = jnp.nan_to_num(jnp.array(reward_buffer), nan=0.0)
    values = jnp.nan_to_num(jnp.array(value_buffer), nan=0.0)
    dones = jnp.nan_to_num(jnp.array(done_buffer), nan=0.0, posinf=1.0, neginf=0.0)

    batch = {
        "observations": observations,
        "actions": actions,
        "log_probs": log_probs,
        "rewards": rewards,
        "values": values,
        "dones": dones,
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
    if applied_action_buffer:
        applied_actions = jnp.stack(applied_action_buffer)
        policy_actions = jnp.stack(action_buffer)
        mpc_actions = jnp.stack(mpc_action_buffer)
        action_norms = jnp.linalg.norm(applied_actions, axis=1)
        policy_norms = jnp.linalg.norm(policy_actions, axis=1)
        mpc_norms = jnp.linalg.norm(mpc_actions, axis=1)
        info_metrics.update(
            {
                "action_norm_mean": float(jnp.mean(action_norms)),
                "action_norm_std": float(jnp.std(action_norms)),
                "policy_action_norm_mean": float(jnp.mean(policy_norms)),
                "mpc_action_norm_mean": float(jnp.mean(mpc_norms)),
                "episode_steps": float(applied_actions.shape[0]),
            }
        )
    if len(value_buffer) > 1:
        value_array = jnp.nan_to_num(jnp.array(value_buffer[:-1]), nan=0.0)
        info_metrics["value_mean"] = float(jnp.mean(value_array))
        info_metrics["value_std"] = float(jnp.std(value_array))
    if mpc_loss_values:
        loss_mean = float(np.mean(mpc_loss_values))
        info_metrics["mpc_loss_mean"] = loss_mean
        info_metrics["mpc_loss_last"] = float(mpc_loss_values[-1])
        info_metrics["mpc_grad_norm_mean"] = float(np.mean(mpc_grad_norms))
        info_metrics["mpc_iterations_mean"] = float(np.mean(mpc_iterations))
        info_metrics["mpc_saturation_mean"] = float(np.mean(mpc_saturation))
        decay = float(np.clip(config.mpc_loss_ema_decay, 0.0, 1.0))
        if not math.isfinite(state.mpc_loss_ema):
            state.mpc_loss_ema = loss_mean
        else:
            state.mpc_loss_ema = (1.0 - decay) * state.mpc_loss_ema + decay * loss_mean
        info_metrics["mpc_loss_ema"] = float(state.mpc_loss_ema)
    elif math.isfinite(state.mpc_loss_ema):
        info_metrics["mpc_loss_ema"] = float(state.mpc_loss_ema)
    info_metrics["action_blend_policy_weight"] = float(policy_weight)
    info_metrics["action_blend_mpc_weight"] = float(mpc_weight)
    info_metrics["action_blend_policy_weight_raw"] = float(policy_weight_raw)
    info_metrics["action_blend_mpc_weight_raw"] = float(mpc_weight_raw)
    if mpc_weight_raw > 1e-6:
        info_metrics["action_blend_mpc_backoff_scale"] = float(mpc_weight / mpc_weight_raw)
    info_metrics["action_blend_progress"] = stage_progress
    info_metrics["reward_scale"] = reward_scale
    if raw_reward_buffer:
        raw_rewards = jnp.nan_to_num(jnp.array(raw_reward_buffer), nan=0.0)
        info_metrics["reward_raw_mean"] = float(jnp.mean(raw_rewards))
        info_metrics["reward_raw_std"] = float(jnp.std(raw_rewards))
    if normalised_reward_buffer:
        norm_rewards = jnp.nan_to_num(jnp.array(normalised_reward_buffer), nan=0.0)
        info_metrics["reward_normalised_mean"] = float(jnp.mean(norm_rewards))
        info_metrics["reward_normalised_std"] = float(jnp.std(norm_rewards))
    if policy_action_resets:
        info_metrics["policy_action_resets"] = float(policy_action_resets)
    if mpc_action_resets:
        info_metrics["mpc_action_resets"] = float(mpc_action_resets)
    if blended_action_resets:
        info_metrics["blended_action_resets"] = float(blended_action_resets)
    state.last_mpc_plan = cached_plan
    return batch, info_metrics


def _maybe_update_adaptive_lr(state: TrainingState, config: PpoEvolutionConfig) -> None:
    if not config.adaptive_lr_enabled:
        return
    approx_kl = state.last_aux.get("approx_kl")
    if approx_kl is None or math.isnan(approx_kl):
        return
    target = max(config.adaptive_lr_target_kl, 1e-5)
    lower = target * config.adaptive_lr_lower_ratio
    upper = target * config.adaptive_lr_upper_ratio
    direction = 0
    if approx_kl > upper:
        direction = -1
    elif approx_kl < lower:
        direction = 1
    else:
        state.lr_adjust_last_direction = 0
        if state.lr_adjust_cooldown > 0:
            state.lr_adjust_cooldown = max(state.lr_adjust_cooldown - 1, 0)
        return

    cooldown_limit = max(int(config.adaptive_lr_cooldown_episodes), 0)
    if (
        cooldown_limit > 0
        and state.lr_adjust_cooldown > 0
        and direction == state.lr_adjust_last_direction
        and direction != 0
    ):
        state.lr_adjust_cooldown = max(state.lr_adjust_cooldown - 1, 0)
        return

    scale = state.lr_scale
    if direction < 0:
        scale *= config.adaptive_lr_decrease_factor
    else:
        scale *= config.adaptive_lr_increase_factor

    clipped_scale = float(np.clip(scale, config.adaptive_lr_min_scale, config.adaptive_lr_max_scale))
    if math.isclose(clipped_scale, state.lr_scale, rel_tol=1e-6, abs_tol=1e-9):
        state.lr_adjust_last_direction = direction
        if state.lr_adjust_cooldown > 0:
            state.lr_adjust_cooldown = max(state.lr_adjust_cooldown - 1, 0)
        return

    state.lr_scale = clipped_scale
    state.lr_adjust_last_direction = direction
    state.lr_adjust_cooldown = cooldown_limit


def _update_stage_progress(
    state: TrainingState,
    curriculum: List[CurriculumStage],
    stage: CurriculumStage,
    episode_return: float,
    logger: logging.Logger,
    config: PpoEvolutionConfig,
) -> Tuple[CurriculumStage, Dict[str, float]]:
    state.stage_episode += 1
    alpha = float(np.clip(config.curriculum_reward_smoothing, 0.0, 1.0))
    if state.stage_episode == 1:
        state.stage_reward_ema = episode_return
    else:
        state.stage_reward_ema = (1.0 - alpha) * state.stage_reward_ema + alpha * episode_return

    if config.curriculum_adaptation and stage.reward_threshold is not None:
        if state.stage_reward_ema >= stage.reward_threshold:
            state.stage_success_counter += 1
        else:
            state.stage_success_counter = max(state.stage_success_counter - 1, 0)
    else:
        state.stage_success_counter = 0

    metrics = {
        "stage_episode": float(state.stage_episode),
        "stage_reward_ema": float(state.stage_reward_ema),
        "stage_success_counter": float(state.stage_success_counter),
    }

    should_advance = False
    min_required = max(stage.min_episodes, 1)
    if config.curriculum_adaptation and stage.reward_threshold is not None:
        if state.stage_episode >= min_required and state.stage_success_counter >= stage.success_episodes:
            should_advance = True
    if state.stage_episode >= stage.episodes:
        should_advance = True

    if not state.lr_schedule_active:
        reward_gate = stage.reward_threshold if stage.reward_threshold is not None else config.curriculum_reward_smoothing
        min_episodes_for_lr = max(stage.min_episodes, 1)
        if (
            state.stage_episode >= min_episodes_for_lr
            and (reward_gate is None or state.stage_reward_ema >= reward_gate)
        ):
            state.lr_schedule_active = True

    if should_advance and state.stage_index < len(curriculum) - 1:
        previous_stage = stage.name
        state.stage_index += 1
        state.stage_episode = 0
        state.stage_success_counter = 0
        state.stage_reward_ema = 0.0
        state.last_mpc_plan = None
        state.best_return = -float("inf")
        state.best_stage_reward = -float("inf")
        state.plateau_counter = 0
        state.last_improvement_episode = 0
        stage = curriculum[state.stage_index]
        state.current_stage_name = stage.name
        state.lr_schedule_active = False
        metrics["stage_transition"] = 1.0
        logger.info("Advancing curriculum from %s to %s", previous_stage, stage.name)
        if config.adaptive_lr_enabled:
            bias = config.stage_lr_bias.get(stage.name)
            if bias is not None:
                state.lr_scale = float(np.clip(bias, config.adaptive_lr_min_scale, config.adaptive_lr_max_scale))
    else:
        metrics["stage_transition"] = 0.0
        state.current_stage_name = stage.name

    return stage, metrics


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
    advantages = jnp.nan_to_num(advantages, nan=0.0)
    returns = jnp.nan_to_num(returns, nan=0.0)

    dataset = {
        "obs": batch["observations"],
        "actions": batch["actions"],
        "log_probs": batch["log_probs"],
        "advantages": advantages,
        "returns": returns,
        "value_preds": batch["values"][:-1],
    }

    base_entropy_coef = float(config.entropy_coef)
    entropy_scale = float(state.entropy_scale)
    entropy_coef = base_entropy_coef * entropy_scale
    entropy_coef = float(
        np.clip(
            entropy_coef,
            max(float(config.entropy_coef_min), 0.0),
            max(float(config.entropy_coef_max), max(float(config.entropy_coef_min), 0.0) + 1e-9),
        )
    )

    def loss_fn(params, minibatch):
        mean, log_std, values = funcs.distribution(params, minibatch["obs"], key=None, deterministic=True)
        log_prob = jax.vmap(_gaussian_log_prob, in_axes=(0, None, 0))(mean, log_std, minibatch["actions"])
        log_prob = jnp.nan_to_num(log_prob, nan=-1e6, posinf=0.0, neginf=-1e6)
        ratio = jnp.exp(log_prob - minibatch["log_probs"])
        ratio = jnp.nan_to_num(ratio, nan=1.0, posinf=1e6, neginf=0.0)
        ratio = jnp.clip(ratio, 1e-6, 1e6)
        clipped = jnp.clip(ratio, 1.0 - config.clip_epsilon, 1.0 + config.clip_epsilon)
        surrogate = ratio * minibatch["advantages"]
        clipped_surrogate = clipped * minibatch["advantages"]
        actor_loss = -jnp.mean(jnp.minimum(surrogate, clipped_surrogate))
        value_clipped = minibatch["value_preds"] + jnp.clip(
            values - minibatch["value_preds"],
            -config.value_clip_epsilon,
            config.value_clip_epsilon,
        )
        value_loss_unclipped = (minibatch["returns"] - values) ** 2
        value_loss_clipped = (minibatch["returns"] - value_clipped) ** 2
        value_loss = jnp.mean(jnp.maximum(value_loss_unclipped, value_loss_clipped))
        entropy = jnp.mean(
            jnp.sum(log_std + 0.5 * jnp.log(2.0 * jnp.pi * jnp.e), axis=-1)
        )
        entropy = jnp.nan_to_num(entropy, nan=0.0)
        approx_kl = jnp.mean(minibatch["log_probs"] - log_prob)
        approx_kl = jnp.nan_to_num(approx_kl, nan=0.0)
        approx_kl = jnp.maximum(approx_kl, 0.0)
        clip_fraction = jnp.mean((jnp.abs(ratio - 1.0) > config.clip_epsilon).astype(jnp.float32))
        clip_fraction = jnp.nan_to_num(clip_fraction, nan=0.0)
        total_loss = actor_loss + config.value_coef * value_loss - entropy_coef * entropy
        return total_loss, (entropy, approx_kl, actor_loss, value_loss, clip_fraction)

    params = state.params
    opt_state = state.opt_state
    num_samples = dataset["obs"].shape[0]

    metrics_sums = {"loss": 0.0, "entropy": 0.0, "approx_kl": 0.0, "policy_loss": 0.0, "value_loss": 0.0, "clip_fraction": 0.0}
    total_batches = 0
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)

    for _ in range(config.num_epochs):
        permutation = np.random.permutation(num_samples)
        for start in range(0, num_samples, config.minibatch_size):
            idx = permutation[start : start + config.minibatch_size]
            minibatch = {k: v[idx] for k, v in dataset.items()}
            (loss_value, aux_vals), grads = grad_fn(params, minibatch)
            updates, opt_state = optimiser.update(grads, opt_state, params)
            scaled_updates = jtu.tree_map(lambda u: state.lr_scale * u, updates)
            params = optax.apply_updates(params, scaled_updates)
            state.update_step += 1
            entropy_val, approx_kl_val, actor_loss_val, value_loss_val, clip_fraction_val = aux_vals
            metrics_sums["loss"] += float(loss_value)
            metrics_sums["entropy"] += float(entropy_val)
            metrics_sums["approx_kl"] += float(approx_kl_val)
            metrics_sums["policy_loss"] += float(actor_loss_val)
            metrics_sums["value_loss"] += float(value_loss_val)
            metrics_sums["clip_fraction"] += float(clip_fraction_val)
            total_batches += 1

    state.params = params
    state.opt_state = opt_state
    if total_batches > 0:
        for key in metrics_sums:
            metrics_sums[key] /= total_batches

    metrics_sums["entropy_coef"] = entropy_coef

    entropy_target = float(config.entropy_target)
    can_adapt_entropy = (
        base_entropy_coef > 0.0
        and float(config.entropy_coef_max) > float(config.entropy_coef_min)
    )
    if entropy_target > 0.0 and can_adapt_entropy and math.isfinite(metrics_sums["entropy"]):
        observed_entropy = metrics_sums["entropy"]
        if math.isfinite(observed_entropy):
            error = entropy_target - observed_entropy
            adjust_speed = max(float(config.entropy_adjust_speed), 0.0)
            if adjust_speed > 0.0:
                adjust_factor = math.exp(adjust_speed * error)
                denom = max(base_entropy_coef, 1e-6)
                min_scale = float(config.entropy_coef_min) / denom
                max_scale = float(config.entropy_coef_max) / denom
                min_scale = max(min_scale, 1e-6)
                max_scale = max(max_scale, min_scale + 1e-6)
                new_scale = float(np.clip(state.entropy_scale * adjust_factor, min_scale, max_scale))
                state.entropy_scale = new_scale
    metrics_sums["entropy_scale"] = state.entropy_scale

    state.last_aux = {
        **metrics_sums,
        "advantage_std": float(jnp.std(advantages)),
        "advantage_mean": float(jnp.mean(advantages)),
        "return_mean": float(jnp.mean(returns)),
        "return_std": float(jnp.std(returns)),
        "entropy_scale": state.entropy_scale,
    }
    return state


def _evolutionary_update(
    env: Tvc2DEnv,
    stage: CurriculumStage,
    state: TrainingState,
    funcs,
    config: PpoEvolutionConfig,
    optimiser: optax.GradientTransformation,
    baseline_reward: float,
) -> Tuple[TrainingState, float, float, bool]:
    if not config.use_evolution or config.population_size <= 1 or config.elite_keep <= 0:
        state.elites = []
        return state, baseline_reward, baseline_reward, False

    rng = state.rng
    population = [state.params] + list(state.elites)
    while len(population) < config.population_size:
        rng, key = jax.random.split(rng)
        population.append(mutate_parameters(key, state.params, config.mutation_scale))

    scores: List[float] = []
    for candidate in population:
        reward = _evaluate_candidate(env, stage, candidate, funcs, config, state.obs_rms)
        scores.append(reward)

    if not scores:
        state.elites = []
        state.rng = rng
        return state, baseline_reward, baseline_reward, False

    elite_keep = max(1, min(config.elite_keep, len(population)))
    elite_indices = np.argsort(scores)[-elite_keep:][::-1]
    elites = [population[i] for i in elite_indices]
    state.elites = elites
    state.rng = rng

    best_reward = float(np.max(scores))
    mean_reward = float(np.mean(scores))

    adopted = False
    best_candidate = elites[0] if elites else None
    if (
        best_candidate is not None
        and best_reward > baseline_reward + config.evolution_adoption_margin
    ):
        state.params = best_candidate
        state.opt_state = optimiser.init(best_candidate)
        adopted = True

    return state, best_reward, mean_reward, adopted


def _evaluate_candidate(
    env: Tvc2DEnv,
    stage: CurriculumStage,
    params,
    funcs,
    config: PpoEvolutionConfig,
    normalizer: RunningNormalizer,
    *,
    episodes: Optional[int] = None,
    max_steps: Optional[int] = None,
) -> float:
    total = 0.0
    base_normalizer = RunningNormalizer(
        count=float(normalizer.count),
        mean=jnp.array(normalizer.mean, copy=True),
        m2=jnp.array(normalizer.m2, copy=True),
    )
    eval_episodes = episodes if episodes is not None else config.evaluation_episodes
    eval_max_steps = max_steps if max_steps is not None else config.evaluation_max_steps
    reward_scale = float(config.reward_scale)
    reward_scale = reward_scale if reward_scale > 0.0 else 1.0
    env_limit = float(getattr(env, "ctrl_limit", config.mpc_config.control_limit))
    action_limit = float(min(env_limit, config.mpc_config.control_limit))
    policy_weight, mpc_weight, _ = _resolve_action_blend_weights(
        config,
        stage,
        stage_episode=config.action_blend_transition_episodes,
        progress_override=1.0,
    )
    mpc_interval = max(1, int(config.mpc_interval))
    for _ in range(eval_episodes):
        if hasattr(env, "configure_stage"):
            env.configure_stage(stage)
        obs = jnp.asarray(env.reset(disturbance_scale=stage.disturbance_scale), dtype=jnp.float32)
        local_rms = _update_normalizer(base_normalizer, obs)
        norm_obs = _normalize_observation(local_rms, obs)
        done = False
        steps = 0
        cached_plan = None
        cached_mpc_action = jnp.zeros(2, dtype=jnp.float32)
        while not done and steps < eval_max_steps:
            if steps % mpc_interval == 0:
                rocket_state = _gather_state(env)
                cached_mpc_action, _, cached_plan = compute_tvc_mpc_action(
                    rocket_state,
                    stage.target_state,
                    config.params,
                    config.mpc_config,
                    warm_start=cached_plan,
                )
            policy_action = funcs.actor(params, norm_obs, None, True)
            blended_action = policy_weight * policy_action + mpc_weight * cached_mpc_action
            action = jnp.clip(
                blended_action,
                -action_limit,
                action_limit,
            )
            result = env.step(np.array(action))
            result_info = result.info
            raw_reward = float(result_info.get("raw_reward", result.reward))
            total += raw_reward * reward_scale
            obs = jnp.asarray(result.observation, dtype=jnp.float32)
            local_rms = _update_normalizer(local_rms, obs)
            norm_obs = _normalize_observation(local_rms, obs)
            done = result.done
            steps += 1
        base_normalizer = local_rms
    return total / max(1, eval_episodes)


def save_training_artifacts(
    history: List[Dict[str, Any]],
    output_dir: Path,
    config: PpoEvolutionConfig,
    pretraining_metrics: Dict[str, Any] | None = None,
) -> None:
    """Persists per-episode and optional pretraining metrics to ``output_dir``."""

    if not history and pretraining_metrics is None:
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    if history:
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

        try:
            import matplotlib

            matplotlib.use("Agg", force=True)
            import matplotlib.pyplot as plt
        except ImportError as exc:  # pragma: no cover - optional dependency
            logging.getLogger("tvc.training").warning("matplotlib unavailable, skipping plots: %s", exc)
        else:
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

    config_path = output_dir / "config.json"
    with config_path.open("w", encoding="utf-8") as config_file:
        json.dump(_config_to_serialisable(config), config_file, indent=2)

    if pretraining_metrics:
        pretraining_path = output_dir / "pretraining_metrics.json"
        pretraining_path.write_text(
            json.dumps(_ensure_json_serialisable(pretraining_metrics), indent=2),
            encoding="utf-8",
        )


def _ensure_json_serialisable(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return float(value)
    if isinstance(value, (np.ndarray, jnp.ndarray)):
        return np.asarray(value).tolist()
    if isinstance(value, dict):
        return {k: _ensure_json_serialisable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_ensure_json_serialisable(v) for v in value]
    return value


def save_policy_checkpoints(state: TrainingState, output_dir: Path) -> Dict[str, Path]:
    """Persists the trained policy parameters, elites, and normaliser statistics."""

    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts: Dict[str, Path] = {}

    final_params = jax.device_get(state.params)
    final_path = output_dir / "policy_final.msgpack"
    final_path.write_bytes(serialization.to_bytes(final_params))
    artifacts["policy_final"] = final_path

    if state.elites:
        best_elite = jax.device_get(state.elites[0])
        elite_path = output_dir / "policy_elite_best.msgpack"
        elite_path.write_bytes(serialization.to_bytes(best_elite))
        artifacts["policy_elite_best"] = elite_path

    mean = np.asarray(jax.device_get(state.obs_rms.mean), dtype=np.float32)
    m2 = np.asarray(jax.device_get(state.obs_rms.m2), dtype=np.float32)
    count = np.asarray([state.obs_rms.count], dtype=np.float32)
    normaliser_path = output_dir / "observation_normalizer.npz"
    np.savez(normaliser_path, count=count, mean=mean, m2=m2)
    artifacts["observation_normalizer"] = normaliser_path

    best_episode = max(state.history, key=lambda item: item.get("rollout_return", -float("inf")), default=None)
    metadata: Dict[str, Any] = {
        "episodes": len(state.history),
        "best_return": float(state.best_return),
        "current_stage": state.current_stage_name,
        "lr_scale": state.lr_scale,
    }
    if best_episode is not None:
        metadata["best_episode"] = best_episode
    if state.pretraining_metrics:
        metadata["pretraining_metrics"] = _ensure_json_serialisable(state.pretraining_metrics)

    metadata_path = output_dir / "policy_metadata.json"
    metadata_path.write_text(json.dumps(_ensure_json_serialisable(metadata), indent=2), encoding="utf-8")
    artifacts["policy_metadata"] = metadata_path

    return artifacts


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
        "adaptive_lr_enabled": config.adaptive_lr_enabled,
        "adaptive_lr_target_kl": config.adaptive_lr_target_kl,
        "adaptive_lr_lower_ratio": config.adaptive_lr_lower_ratio,
        "adaptive_lr_upper_ratio": config.adaptive_lr_upper_ratio,
        "adaptive_lr_increase_factor": config.adaptive_lr_increase_factor,
        "adaptive_lr_decrease_factor": config.adaptive_lr_decrease_factor,
        "adaptive_lr_min_scale": config.adaptive_lr_min_scale,
        "adaptive_lr_max_scale": config.adaptive_lr_max_scale,
        "stage_lr_bias": dict(config.stage_lr_bias),
        "curriculum_adaptation": config.curriculum_adaptation,
        "curriculum_reward_smoothing": config.curriculum_reward_smoothing,
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
