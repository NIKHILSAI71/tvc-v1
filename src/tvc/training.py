"""Enhanced PPO + Evolution training for 3D TVC with real-time visualization."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple

import jax
import jax.numpy as jnp
import mujoco.viewer
import numpy as np
import optax
from flax import serialization
from jax import Array

from .curriculum import CurriculumStage, build_curriculum
from .dynamics import RocketParams
from .env import TvcEnv
from .mpc import MpcConfig
from .policies import PolicyConfig, PolicyFunctions, build_policy_network, mutate_parameters

LOGGER = logging.getLogger(__name__)


@dataclass
class RunningNormalizer:
    """Welford's online normalization."""
    count: float = 0.0
    mean: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=np.float32))
    m2: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=np.float32))

    @classmethod
    def initialise(cls, observation: np.ndarray) -> "RunningNormalizer":
        obs = np.asarray(observation, dtype=np.float32)
        return cls(count=1.0, mean=obs, m2=np.zeros_like(obs))

    def normalize(self, observation: np.ndarray) -> jnp.ndarray:
        obs = np.asarray(observation, dtype=np.float32)
        if self.mean.size == 0 or self.mean.shape != obs.shape or self.count < 1.0:
            self.count = 1.0
            self.mean = obs.copy()
            self.m2 = np.zeros_like(obs)
            return jnp.asarray(obs, dtype=jnp.float32)

        self.count += 1.0
        delta = obs - self.mean
        self.mean = self.mean + delta / self.count
        delta2 = obs - self.mean
        self.m2 = self.m2 + delta * delta2
        variance = np.maximum(self.m2 / self.count, 1e-6)
        normalized = (obs - self.mean) / np.sqrt(variance)
        return jnp.asarray(normalized, dtype=jnp.float32)


@dataclass
class TrainingState:
    """PPO + Evolution training state."""
    params: Any
    opt_state: optax.OptState
    rng: jax.Array
    obs_rms: RunningNormalizer
    history: List[Dict[str, Any]] = field(default_factory=list)
    update_step: int = 0
    stage_index: int = 0
    stage_episode: int = 0
    best_return: float = -float("inf")
    elites: List[Any] = field(default_factory=list)  # For evolution
    moving_avg_return: float = 0.0
    moving_avg_alpha: float = 0.1
    # Success tracking
    total_successes: int = 0
    stage_successes: int = 0
    stage_attempts: int = 0
    # Rolling success rate tracking (window of 20 episodes)
    recent_successes: List[bool] = field(default_factory=lambda: [False] * 20)

    def update_rolling_success(self, success: bool) -> float:
        """Update rolling success rate window."""
        self.recent_successes.pop(0)
        self.recent_successes.append(success)
        return sum(self.recent_successes) / len(self.recent_successes)


@dataclass(frozen=True)
class TrainingConfig:
    """Enhanced PPO + Neuroevolution configuration.

    Based on successful rocket stabilization system:
    - Large population (20+ candidates)
    - Score-based selection with top performers
    - Mutation of best candidates to create next generation
    - Combined with PPO gradient-based learning
    """
    gamma: float = 0.99
    lam: float = 0.95
    learning_rate: float = 5e-4  # Increased from 3e-4 for faster learning
    clip_epsilon: float = 0.2    # Standard PPO value
    rollout_length: int = 3072   # Increased from 2048 for better value estimation
    num_epochs: int = 4          # Keep at 4 to prevent overfitting
    minibatch_size: int = 256    # Increased from 128 for more stable gradients
    value_clip_epsilon: float = 0.2  # Match actor clipping
    grad_clip_norm: float = 0.5  # Conservative to prevent instability
    entropy_coef: float = 0.01   # Initial exploration coefficient
    entropy_coef_decay: float = 0.9995
    value_coef: float = 0.5      # Balanced value function importance
    weight_decay: float = 1e-5   # Light regularization

    # Neuroevolution settings - Optimized for better exploration
    use_evolution: bool = True
    population_size: int = 10  # Increased from 5 for better diversity
    elite_keep: int = 2  # Keep top 2 performers
    mutation_scale: float = 0.08  # Increased from 0.05 for more exploration
    mutation_prob: float = 0.8  # 80% of parameters mutated
    evolution_interval: int = 3  # More frequent evolution (from 5)
    fitness_episodes: int = 1  # Single episode evaluation for speed

    # Configuration objects
    policy_config: PolicyConfig = PolicyConfig()
    mpc_config: MpcConfig = MpcConfig()
    rocket_params: RocketParams = RocketParams()


def _compute_gae(
    rewards: jnp.ndarray,
    values: jnp.ndarray,
    dones: jnp.ndarray,
    gamma: float,
    lam: float,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Compute Generalized Advantage Estimation."""
    def scan_fn(carry: jnp.ndarray, inputs: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]) -> Tuple[jnp.ndarray, jnp.ndarray]:
        reward, value, done = inputs
        next_value = carry
        delta = reward + gamma * next_value * (1.0 - done) - value
        advantage = delta + gamma * lam * (1.0 - done) * carry
        return advantage, advantage

    _, advantages_rev = jax.lax.scan(
        scan_fn,
        jnp.asarray(0.0, dtype=jnp.float32),
        (rewards, values[:-1], dones),
        reverse=True,
    )
    advantages = advantages_rev[::-1]
    returns = advantages + values[:-1]
    return advantages, returns


def _gaussian_log_prob(mean: jnp.ndarray, log_std: jnp.ndarray, actions: jnp.ndarray) -> jnp.ndarray:
    """Compute Gaussian log probability."""
    std = jnp.exp(log_std)
    var = std * std
    log_density = -0.5 * jnp.sum(jnp.square(actions - mean) / var + 2.0 * log_std + jnp.log(2.0 * jnp.pi), axis=-1)
    return log_density


def _collect_rollout(
    env: TvcEnv,
    stage: CurriculumStage,
    state: TrainingState,
    funcs: PolicyFunctions,
    config: TrainingConfig,
    normalize_rewards: bool = True,
    viewer: mujoco.viewer.Viewer | None = None,
) -> Tuple[Dict[str, jnp.ndarray], Dict[str, float]]:
    """Collect rollout trajectory with optional reward normalization."""
    env.configure_stage(stage)
    observation = env.reset()

    obs_buffer = []
    action_buffer = []
    logprob_buffer = []
    reward_buffer = []
    value_buffer = []
    done_buffer = []

    obs_rms = state.obs_rms
    norm_observation = obs_rms.normalize(observation)

    # Track each episode's success within the rollout
    episode_successes = []
    episode_returns = []
    current_episode_return = 0.0
    current_episode_steps = 0

    for step in range(config.rollout_length):
        state.rng, policy_key = jax.random.split(state.rng)

        mean, log_std, value = funcs.distribution(
            state.params,
            norm_observation,
            key=None,
            deterministic=True,
        )

        std = jnp.exp(log_std)
        epsilon = jax.random.normal(policy_key, shape=mean.shape)
        action = mean + std * epsilon
        action = jnp.clip(action, -10.0, 10.0)  # Safety clip

        log_prob = _gaussian_log_prob(mean, log_std, action)

        obs_buffer.append(norm_observation)
        action_buffer.append(action)
        logprob_buffer.append(log_prob)
        value_buffer.append(value)

        action_np = np.asarray(action, dtype=np.float32)
        step_result = env.step(action_np)

        if viewer:
            viewer.sync()

        reward_buffer.append(step_result.reward)
        done_buffer.append(1.0 if step_result.done else 0.0)

        current_episode_return += step_result.reward
        current_episode_steps += 1

        # Check success at episode termination
        if step_result.done:
            pos = env.data.qpos[0:3]
            vel = env.data.qvel[0:3]
            quat = env.data.qpos[3:7]
            omega = env.data.qvel[3:6]
            target_pos = np.array(stage.target_position, dtype=np.float32)
            target_vel = np.array(stage.target_velocity, dtype=np.float32)
            target_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

            pos_error = float(np.linalg.norm(pos - target_pos))
            vel_error = float(np.linalg.norm(vel - target_vel))
            orient_alignment = float(np.abs(np.dot(quat, target_quat)))
            angular_vel_mag = float(np.linalg.norm(omega))

            # Evaluate success at episode end - MUST be stable and upright
            episode_success = (
                pos_error < stage.position_tolerance and
                vel_error < stage.velocity_tolerance and
                orient_alignment > 0.98 and  # Stricter: within ~11 degrees
                angular_vel_mag < stage.angular_velocity_tolerance  # Must not be spinning
            )

            episode_successes.append(episode_success)
            episode_returns.append(current_episode_return)

            # Reset episode tracking
            current_episode_return = 0.0
            current_episode_steps = 0

            observation = env.reset()
            norm_observation = obs_rms.normalize(observation)
        else:
            norm_observation = obs_rms.normalize(step_result.observation)

    # If rollout ended mid-episode, evaluate current state
    if current_episode_steps > 0:
        pos = env.data.qpos[0:3]
        vel = env.data.qvel[0:3]
        quat = env.data.qpos[3:7]
        target_pos = np.array(stage.target_position, dtype=np.float32)
        target_vel = np.array(stage.target_velocity, dtype=np.float32)
        target_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

        pos_error = float(np.linalg.norm(pos - target_pos))
        vel_error = float(np.linalg.norm(vel - target_vel))
        orient_alignment = float(np.abs(np.dot(quat, target_quat)))

        episode_success = (
            pos_error < stage.position_tolerance and
            vel_error < stage.velocity_tolerance and
            orient_alignment > 0.95
        )

        episode_successes.append(episode_success)
        episode_returns.append(current_episode_return)
    else:
        # Use last completed episode if rollout ended exactly on episode boundary
        pos = env.data.qpos[0:3]
        vel = env.data.qvel[0:3]
        quat = env.data.qpos[3:7]
        target_pos = np.array(stage.target_position, dtype=np.float32)
        target_vel = np.array(stage.target_velocity, dtype=np.float32)

        pos_error = float(np.linalg.norm(pos - target_pos))
        vel_error = float(np.linalg.norm(vel - target_vel))

    # Final value for bootstrap
    _, _, final_value = funcs.distribution(
        state.params,
        norm_observation,
        key=None,
        deterministic=True,
    )
    value_buffer.append(final_value)

    rewards_array = jnp.array(reward_buffer, dtype=jnp.float32)

    # Normalize rewards to prevent value function scale issues
    if normalize_rewards and len(reward_buffer) > 1:
        reward_mean = jnp.mean(rewards_array)
        reward_std = jnp.std(rewards_array) + 1e-8
        normalized_rewards = (rewards_array - reward_mean) / reward_std
    else:
        normalized_rewards = rewards_array

    batch = {
        "observations": jnp.stack(obs_buffer),
        "actions": jnp.stack(action_buffer),
        "log_probs": jnp.stack(logprob_buffer),
        "rewards": normalized_rewards,  # Use normalized rewards
        "values": jnp.stack(value_buffer),
        "dones": jnp.array(done_buffer, dtype=jnp.float32),
    }

    # Aggregate success across all episodes in rollout
    overall_success = any(episode_successes) if episode_successes else False
    num_episodes = len(episode_successes)
    num_successful = sum(episode_successes)
    success_rate = num_successful / num_episodes if num_episodes > 0 else 0.0

    stats = {
        "episode_return": float(jnp.sum(rewards_array)),  # Total return across all episodes in rollout
        "reward_mean": float(jnp.mean(rewards_array)),
        "reward_std": float(jnp.std(rewards_array)),
        "value_mean": float(jnp.mean(batch["values"][:-1])),
        "action_mean": float(jnp.mean(jnp.abs(batch["actions"]))),
        "reward_normalized": normalize_rewards,
        "episode_success": overall_success,  # True if ANY episode succeeded
        "final_position_error": pos_error,
        "final_velocity_error": vel_error,
        "final_orientation_alignment": orient_alignment,
        "num_episodes_in_rollout": num_episodes,
        "num_successful_episodes": num_successful,
        "rollout_success_rate": success_rate,
    }

    return batch, stats


def _ppo_update(
    state: TrainingState,
    batch: Dict[str, jnp.ndarray],
    optimizer: optax.GradientTransformation,
    funcs: PolicyFunctions,
    config: TrainingConfig,
    current_entropy_coef: float,
) -> Tuple[TrainingState, Dict[str, float]]:
    """Enhanced PPO update with better logging."""
    advantages, returns = _compute_gae(
        batch["rewards"],
        batch["values"],
        batch["dones"],
        config.gamma,
        config.lam,
    )

    # Normalize advantages
    advantages = (advantages - jnp.mean(advantages)) / (jnp.std(advantages) + 1e-8)
    advantages = jnp.nan_to_num(advantages, nan=0.0)

    # Normalize returns
    returns = (returns - jnp.mean(returns)) / (jnp.std(returns) + 1e-8)
    returns = jnp.nan_to_num(returns, nan=0.0)

    dataset = {
        "obs": batch["observations"],
        "actions": batch["actions"],
        "log_probs": batch["log_probs"],
        "advantages": advantages,
        "returns": returns,
        "value_preds": batch["values"][:-1],
    }

    def loss_fn(params: Any, minibatch: Dict[str, jnp.ndarray]) -> Tuple[jnp.ndarray, Tuple]:
        # Vectorized policy evaluation over batch
        def single_forward(obs):
            return funcs.distribution(params, obs, key=None, deterministic=True)

        mean, log_std, values = jax.vmap(single_forward)(minibatch["obs"])
        log_prob = jax.vmap(_gaussian_log_prob)(mean, log_std, minibatch["actions"])
        log_prob = jnp.nan_to_num(log_prob, nan=-1e6)

        ratio = jnp.exp(log_prob - minibatch["log_probs"])
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
        value_loss_unclipped = jnp.square(minibatch["returns"] - values)
        value_loss_clipped = jnp.square(minibatch["returns"] - value_clipped)
        value_loss = jnp.mean(jnp.maximum(value_loss_unclipped, value_loss_clipped))

        entropy = jnp.mean(jnp.sum(log_std + 0.5 * jnp.log(2.0 * jnp.pi * jnp.e), axis=-1))
        entropy = jnp.nan_to_num(entropy, nan=0.0)
        entropy = jnp.clip(entropy, -10.0, 10.0)  # Prevent extreme values

        approx_kl = jnp.mean(minibatch["log_probs"] - log_prob)
        approx_kl = jnp.nan_to_num(approx_kl, nan=0.0)
        approx_kl = jnp.clip(approx_kl, 0.0, 10.0)  # KL must be non-negative

        clip_frac = jnp.mean((jnp.abs(ratio - 1.0) > config.clip_epsilon).astype(jnp.float32))

        total_loss = actor_loss + config.value_coef * value_loss - current_entropy_coef * entropy
        return total_loss, (entropy, approx_kl, actor_loss, value_loss, clip_frac, ratio)

    params = state.params
    opt_state = state.opt_state

    total_actor_loss = 0.0
    total_value_loss = 0.0
    total_entropy = 0.0
    total_kl = 0.0
    total_clip_frac = 0.0
    update_count = 0
    early_stop = False

    # Multiple epochs over the dataset
    for epoch in range(config.num_epochs):
        if early_stop:
            break

        # Shuffle and create minibatches
        num_samples = dataset["obs"].shape[0]
        indices = np.arange(num_samples)
        np.random.shuffle(indices)

        for start in range(0, num_samples, config.minibatch_size):
            end = min(start + config.minibatch_size, num_samples)
            batch_indices = indices[start:end]

            minibatch = {k: v[batch_indices] for k, v in dataset.items()}

            (loss, (entropy, kl, actor_loss, value_loss, clip_frac, ratio)), grads = jax.value_and_grad(
                loss_fn, has_aux=True
            )(params, minibatch)

            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)

            total_actor_loss += float(actor_loss)
            total_value_loss += float(value_loss)
            total_entropy += float(entropy)
            total_kl += float(kl)
            total_clip_frac += float(clip_frac)
            update_count += 1

            # Early stopping if KL divergence gets too high
            # Relaxed from 0.02 to 0.05 to prevent premature policy freezing
            if float(kl) > 0.05 and epoch > 0:  # Allow first epoch to complete
                LOGGER.debug("Early stop at epoch %d, KL=%.4f", epoch + 1, float(kl))
                early_stop = True
                break

    state.params = params
    state.opt_state = opt_state
    state.update_step += 1

    metrics = {
        "loss": float(loss),
        "actor_loss": total_actor_loss / update_count,
        "value_loss": total_value_loss / update_count,
        "entropy": total_entropy / update_count,
        "kl": total_kl / update_count,
        "clip_frac": total_clip_frac / update_count,
        "entropy_coef": current_entropy_coef,
    }

    return state, metrics


def _evaluate_candidate(
    env: TvcEnv,
    stage: CurriculumStage,
    params: Any,
    funcs: PolicyFunctions,
    obs_rms: RunningNormalizer,
    num_episodes: int = 3,
) -> float:
    """Evaluate a policy candidate."""
    total_return = 0.0

    for _ in range(num_episodes):
        env.configure_stage(stage)
        obs = env.reset()
        episode_return = 0.0
        done = False
        steps = 0

        while not done and steps < 500:
            norm_obs = obs_rms.normalize(obs)
            action = funcs.actor(params, norm_obs, key=None, deterministic=True)
            action_np = np.asarray(action, dtype=np.float32)
            result = env.step(action_np)
            episode_return += result.reward
            obs = result.observation
            done = result.done
            steps += 1

        total_return += episode_return

    return total_return / num_episodes


def train_controller(
    total_episodes: int,
    config: TrainingConfig = TrainingConfig(),
    output_dir: Path | None = None,
    seed: int = 42,
    resume_from: Path | None = None,  # Resume from checkpoint
    visualize: bool = False,
) -> TrainingState:
    """Enhanced PPO + Evolution training with visualization."""

    LOGGER.info("=" * 80)
    LOGGER.info("🚀 ENHANCED TVC TRAINING - PPO + Evolution")
    LOGGER.info("=" * 80)
    LOGGER.info("Episodes: %d | Rollout: %d | Evolution: %s",
                total_episodes, config.rollout_length, config.use_evolution)
    LOGGER.info("=" * 80)

    env = TvcEnv(dt=0.02, ctrl_limit=0.14, max_steps=1000, seed=seed)
    env.apply_rocket_params(config.rocket_params)

    # Build curriculum
    curriculum = build_curriculum()
    current_stage = curriculum[0]
    env.configure_stage(current_stage)

    # Initialize policy
    policy_funcs = build_policy_network(config.policy_config)
    rng = jax.random.PRNGKey(seed)
    init_rng, rng = jax.random.split(rng)

    sample_obs = env.reset()
    params = policy_funcs.init(init_rng, sample_obs)
    obs_rms = RunningNormalizer.initialise(sample_obs)

    # Setup optimizer with adaptive learning rate schedule
    # Cosine annealing with warmup for stable learning
    if total_episodes < 10:
        # Use constant learning rate for short runs
        lr_schedule = config.learning_rate
    else:
        warmup_steps = min(50, max(1, total_episodes // 10))  # Adaptive warmup based on total episodes
        total_steps = total_episodes
        decay_steps = max(1, total_steps - warmup_steps)  # Ensure positive decay_steps
        lr_schedule = optax.warmup_cosine_decay_schedule(
            init_value=config.learning_rate * 0.1,  # Start at 10% of target LR
            peak_value=config.learning_rate,  # Peak at target LR
            warmup_steps=warmup_steps,
            decay_steps=decay_steps,
            end_value=config.learning_rate * 0.1,  # End at 10% for fine-tuning
        )

    optimizer = optax.chain(
        optax.clip_by_global_norm(config.grad_clip_norm),
        optax.adamw(
            learning_rate=lr_schedule,
            weight_decay=config.weight_decay,
        ),
    )
    opt_state = optimizer.init(params)

    # Resume from checkpoint if specified
    start_episode = 0
    if resume_from and resume_from.exists():
        LOGGER.info("📂 Resuming from checkpoint: %s", resume_from)
        try:
            with open(resume_from, "rb") as f:
                params = serialization.from_bytes(params, f.read())
            # Try to load training history if available
            history_path = resume_from.parent.parent / "training_history.json"
            if history_path.exists():
                with open(history_path, "r") as f:
                    history = json.load(f)
                start_episode = len(history)
                LOGGER.info("📊 Loaded history: %d episodes completed", start_episode)
        except Exception as e:
            LOGGER.warning("⚠️ Could not resume from checkpoint: %s", e)
            LOGGER.info("Starting fresh training...")

    state = TrainingState(
        params=params,
        opt_state=opt_state,
        rng=rng,
        obs_rms=obs_rms,
    )

    LOGGER.info("✅ Training initialized | Policy params: %d | Start episode: %d",
                sum(p.size for p in jax.tree_util.tree_leaves(params)), start_episode)

    current_entropy_coef = config.entropy_coef
    start_time = time.time()

    # Initialize viewer if visualization is enabled
    viewer = None
    if visualize:
        viewer = mujoco.viewer.launch_passive(env.model, env.data)
        LOGGER.info("🎥 Live visualization enabled - MuJoCo viewer launched")

    # Training loop
    for episode in range(start_episode, total_episodes):
        episode_start = time.time()

        # Select curriculum stage
        stage_idx = min(state.stage_index, len(curriculum) - 1)
        stage = curriculum[stage_idx]

        # Collect rollout
        batch, rollout_stats = _collect_rollout(env, stage, state, policy_funcs, config, viewer=viewer)

        # PPO update
        state, update_metrics = _ppo_update(state, batch, optimizer, policy_funcs, config, current_entropy_coef)

        # Decay entropy coefficient - tied to stage progression for stability
        # Only decay every 10 episodes to prevent collapse
        if state.stage_episode > 0 and state.stage_episode % 10 == 0:
            current_entropy_coef *= config.entropy_coef_decay
            current_entropy_coef = max(current_entropy_coef, 1e-4)
            LOGGER.debug("Entropy coef decayed: %.6f", current_entropy_coef)

        # Neuroevolution step - Population-based training like rocket stabilization
        evolution_metrics = {}
        if config.use_evolution and (episode + 1) % config.evolution_interval == 0:
            state.rng, mut_rng = jax.random.split(state.rng)

            # Evaluate current policy as baseline
            current_fitness = _evaluate_candidate(env, stage, state.params, policy_funcs, state.obs_rms,
                                                 num_episodes=config.fitness_episodes)

            # Build population: current + elites + new mutants
            population = [(current_fitness, state.params)]

            # Add stored elites if available
            for elite in state.elites:
                elite_fitness = _evaluate_candidate(env, stage, elite, policy_funcs, state.obs_rms,
                                                   num_episodes=config.fitness_episodes)
                population.append((elite_fitness, elite))

            # Generate mutants from current best and elites
            mutant_count = config.population_size - len(population)
            for i in range(mutant_count):
                mut_key = jax.random.fold_in(mut_rng, i)

                # Mutate from a random elite or current params
                if state.elites and i % 2 == 0:
                    parent_idx = i % len(state.elites)
                    parent = state.elites[parent_idx]
                else:
                    parent = state.params

                mutant = mutate_parameters(mut_key, parent, config.mutation_scale, config.mutation_prob)
                fitness = _evaluate_candidate(env, stage, mutant, policy_funcs, state.obs_rms,
                                            num_episodes=config.fitness_episodes)
                population.append((fitness, mutant))

            # Sort by fitness (higher is better)
            population.sort(key=lambda x: x[0], reverse=True)

            # Update statistics
            best_fitness = population[0][0]
            mean_fitness = np.mean([f for f, _ in population])
            evolution_metrics["best_fitness"] = best_fitness
            evolution_metrics["mean_fitness"] = mean_fitness
            evolution_metrics["current_fitness"] = current_fitness

            # Adopt best candidate if it's significantly better (lower threshold)
            if best_fitness > current_fitness + 0.5:  # Reduced from 1.0 to 0.5
                state.params = population[0][1]
                LOGGER.info("  🧬 Evolution: Adopted best | Fitness: %.2f → %.2f | Pop mean: %.2f",
                           current_fitness, best_fitness, mean_fitness)
                evolution_metrics["evolution_adopted"] = 1.0
                evolution_metrics["fitness_improvement"] = best_fitness - current_fitness

                # Reset optimizer state for new params
                state.opt_state = optimizer.init(state.params)
            else:
                evolution_metrics["evolution_adopted"] = 0.0
                if episode % 10 == 0:  # Log occasionally when not adopting
                    LOGGER.info("  🧬 Evolution: Kept current | Best: %.2f | Current: %.2f | Mean: %.2f",
                               best_fitness, current_fitness, mean_fitness)

            # Store top elites for next generation
            state.elites = [params for _, params in population[:config.elite_keep]]
            evolution_metrics["elite_fitness"] = [f for f, _ in population[:config.elite_keep]]

        # Logging
        episode_return = rollout_stats["episode_return"]
        episode_success = rollout_stats.get("episode_success", False)
        state.best_return = max(state.best_return, episode_return)
        state.moving_avg_return = (state.moving_avg_alpha * episode_return +
                                   (1 - state.moving_avg_alpha) * state.moving_avg_return)

        # Update success tracking
        state.stage_attempts += 1
        if episode_success:
            state.total_successes += 1
            state.stage_successes += 1

        # Calculate success rates
        rolling_success_rate = state.update_rolling_success(episode_success)
        stage_success_rate = state.stage_successes / max(state.stage_attempts, 1)
        total_success_rate = state.total_successes / max(episode + 1, 1)

        episode_time = time.time() - episode_start
        elapsed = time.time() - start_time

        metrics = {
            "episode": episode + 1,
            "stage": stage.name,
            "return": episode_return,
            "best_return": state.best_return,
            "moving_avg": state.moving_avg_return,
            "episode_time": episode_time,
            "rolling_success_rate": rolling_success_rate,
            "stage_success_rate": stage_success_rate,
            "total_success_rate": total_success_rate,
            **rollout_stats,
            **update_metrics,
            **evolution_metrics,
        }

        state.history.append(metrics)

        # Enhanced logging
        if (episode + 1) % 1 == 0:  # Log every episode
            success_indicator = "✓" if episode_success else "✗"
            LOGGER.info(
                "Ep %3d/%d | Stage: %-20s | %s | Ret: %7.2f | Avg: %7.2f | Best: %7.2f | "
                "Success: %5.1f%% (rolling: %5.1f%%) | PosErr: %.3fm | Loss: %6.3f | Time: %.1fs",
                episode + 1,
                total_episodes,
                stage.name,
                success_indicator,
                episode_return,
                state.moving_avg_return,
                state.best_return,
                stage_success_rate * 100,
                rolling_success_rate * 100,
                rollout_stats.get("final_position_error", 0.0),
                update_metrics["loss"],
                episode_time,
            )

        # Periodic checkpointing (every 50 episodes)
        if output_dir and (episode + 1) % 50 == 0:
            checkpoint_dir = output_dir / "checkpoints"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

            checkpoint_path = checkpoint_dir / f"policy_ep{episode+1:04d}.msgpack"
            with open(checkpoint_path, "wb") as f:
                f.write(serialization.to_bytes(state.params))
            LOGGER.info("💾 Checkpoint saved: %s", checkpoint_path)

            if episode_return >= state.best_return:
                best_path = output_dir / "policy_best.msgpack"
                with open(best_path, "wb") as f:
                    f.write(serialization.to_bytes(state.params))
                LOGGER.info("🏆 New best policy saved: %s (return: %.2f)", best_path, episode_return)

        state.stage_episode += 1

        # Check if we should progress to next stage (performance-based)
        can_progress = (
            state.stage_index < len(curriculum) - 1 and
            state.stage_episode >= stage.min_episodes and  # Minimum episodes completed
            (
                # Either: completed scheduled episodes
                state.stage_episode >= stage.episodes or
                # Or: achieved high success rate early (80%+ over last 20 episodes)
                (rolling_success_rate >= 0.8 and state.stage_episode >= stage.min_episodes)
            )
        )

        if can_progress:
            LOGGER.info("=" * 80)
            LOGGER.info("📈 STAGE COMPLETE: %s | Success Rate: %.1f%% (%d/%d)",
                       stage.name, stage_success_rate * 100,
                       state.stage_successes, state.stage_attempts)
            if rolling_success_rate >= 0.8 and state.stage_episode < stage.episodes:
                LOGGER.info("🚀 EARLY PROGRESSION: Achieved 80%% rolling success rate!")
            LOGGER.info("📈 STAGE PROGRESSION: %s → %s",
                       stage.name, curriculum[state.stage_index + 1].name)
            LOGGER.info("=" * 80)
            state.stage_index += 1
            state.stage_episode = 0
            # Reset stage-specific counters
            state.stage_successes = 0
            state.stage_attempts = 0

    # Save results
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save policy
        policy_path = output_dir / "policy_final.msgpack"
        with open(policy_path, "wb") as f:
            f.write(serialization.to_bytes(state.params))
        LOGGER.info("💾 Saved policy: %s", policy_path)

        # Save history
        history_path = output_dir / "training_history.json"
        with open(history_path, "w") as f:
            json.dump(state.history, f, indent=2)
        LOGGER.info("💾 Saved history: %s", history_path)

        # Generate plots
        try:
            _generate_plots(state.history, output_dir)
            LOGGER.info("📊 Generated training plots")
        except Exception as e:
            LOGGER.warning("Could not generate plots: %s", e)

    total_time = time.time() - start_time
    final_success_rate = state.total_successes / max(total_episodes, 1)
    LOGGER.info("=" * 80)
    LOGGER.info("🎉 TRAINING COMPLETE")
    LOGGER.info("Total time: %.1f min | Best return: %.2f | Final avg: %.2f",
                total_time / 60, state.best_return, state.moving_avg_return)
    LOGGER.info("Overall Success Rate: %.1f%% (%d/%d successful episodes)",
                final_success_rate * 100, state.total_successes, total_episodes)
    LOGGER.info("=" * 80)

    return state


def _generate_plots(history: List[Dict], output_dir: Path) -> None:
    """Generate comprehensive training visualization plots."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
    except ImportError:
        LOGGER.warning("Matplotlib not available, skipping plots")
        return

    episodes = [h["episode"] for h in history]
    returns = [h["return"] for h in history]
    moving_avgs = [h.get("moving_avg", 0) for h in history]
    value_losses = [h["value_loss"] for h in history]
    entropies = [h["entropy"] for h in history]
    kls = [h["kl"] for h in history]
    success_rates = [h.get("stage_success_rate", 0) * 100 for h in history]
    pos_errors = [h.get("final_position_error", 0) for h in history]

    # Main training progress plot
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("TVC Training Progress", fontsize=16, fontweight='bold')

    # Returns
    axes[0, 0].plot(episodes, returns, alpha=0.3, label="Episode Return")
    axes[0, 0].plot(episodes, moving_avgs, linewidth=2, label="Moving Average")
    axes[0, 0].axhline(y=0, color='r', linestyle='--', alpha=0.5)
    axes[0, 0].set_xlabel("Episode")
    axes[0, 0].set_ylabel("Return")
    axes[0, 0].set_title("Returns Over Time")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Success Rate (NEW)
    axes[0, 1].plot(episodes, success_rates, color='green', linewidth=2)
    axes[0, 1].axhline(y=80, color='orange', linestyle='--', alpha=0.5, label="80% target")
    axes[0, 1].set_xlabel("Episode")
    axes[0, 1].set_ylabel("Success Rate (%)")
    axes[0, 1].set_title("Landing Success Rate")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim([0, 105])

    # Position Error (NEW)
    axes[0, 2].plot(episodes, pos_errors, color='red', alpha=0.6, linewidth=1)
    axes[0, 2].set_xlabel("Episode")
    axes[0, 2].set_ylabel("Position Error (m)")
    axes[0, 2].set_title("Final Position Error")
    axes[0, 2].grid(True, alpha=0.3)
    axes[0, 2].set_yscale('log')

    # Value Loss
    axes[1, 0].plot(episodes, value_losses, color='orange')
    axes[1, 0].set_xlabel("Episode")
    axes[1, 0].set_ylabel("Value Loss")
    axes[1, 0].set_title("Value Function Loss")
    axes[1, 0].grid(True, alpha=0.3)

    # Return Distribution
    axes[1, 1].hist(returns, bins=20, alpha=0.7, color='blue', edgecolor='black')
    axes[1, 1].axvline(x=np.mean(returns), color='r', linestyle='--', linewidth=2, label=f"Mean: {np.mean(returns):.2f}")
    axes[1, 1].set_xlabel("Return")
    axes[1, 1].set_ylabel("Frequency")
    axes[1, 1].set_title("Return Distribution")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    # Training Summary (UPDATED with success metrics)
    axes[1, 2].axis('off')
    final_success_rate = success_rates[-1] if success_rates else 0.0
    total_successes = sum(1 for h in history if h.get("episode_success", False))
    summary_text = f"""
    Training Summary
    ================
    Total Episodes: {len(history)}

    Returns:
      Best: {max(returns):.2f}
      Mean: {np.mean(returns):.2f}
      Final Avg: {moving_avgs[-1]:.2f}

    Accuracy:
      Final Success: {final_success_rate:.1f}%
      Total Success: {total_successes}/{len(history)}
      Avg Pos Error: {np.mean(pos_errors):.3f}m

    Final Metrics:
      Value Loss: {value_losses[-1]:.4f}
      Entropy: {entropies[-1]:.3f}
      KL: {kls[-1]:.4f}
    """
    axes[1, 2].text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
                    verticalalignment='center')

    plt.tight_layout()
    plot_path = output_dir / "training_progress.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    LOGGER.info("📊 Plot saved: %s", plot_path)

    # Additional detailed plots
    _generate_detailed_plots(history, output_dir)
    LOGGER.info("📊 Plot saved: %s", output_dir / "training_dynamics.png")

    _generate_evolution_plots(history, output_dir)
    if any("best_fitness" in h for h in history):
        LOGGER.info("📊 Plot saved: %s", output_dir / "evolution_progress.png")

    _generate_reward_analysis_plots(history, output_dir)
    LOGGER.info("📊 Plot saved: %s", output_dir / "reward_analysis.png")

    _generate_learning_dynamics_plots(history, output_dir)
    LOGGER.info("📊 Plot saved: %s", output_dir / "learning_dynamics.png")


def _generate_detailed_plots(history: List[Dict], output_dir: Path) -> None:
    """Generate detailed action and reward component analysis."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    episodes = [h["episode"] for h in history]
    action_means = [h.get("action_mean", 0) for h in history]
    reward_means = [h.get("reward_mean", 0) for h in history]
    reward_stds = [h.get("reward_std", 0) for h in history]
    value_means = [h.get("value_mean", 0) for h in history]
    actor_losses = [h.get("actor_loss", 0) for h in history]
    clip_fracs = [h.get("clip_frac", 0) * 100 for h in history]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Training Dynamics Analysis", fontsize=16, fontweight='bold')

    # Action magnitude
    axes[0, 0].plot(episodes, action_means, color='blue', linewidth=2)
    axes[0, 0].set_xlabel("Episode")
    axes[0, 0].set_ylabel("Mean |Action|")
    axes[0, 0].set_title("Action Magnitude Over Time")
    axes[0, 0].grid(True, alpha=0.3)

    # Reward statistics
    axes[0, 1].plot(episodes, reward_means, color='green', label="Mean Reward", linewidth=2)
    axes[0, 1].fill_between(episodes,
                           np.array(reward_means) - np.array(reward_stds),
                           np.array(reward_means) + np.array(reward_stds),
                           alpha=0.3, color='green')
    axes[0, 1].set_xlabel("Episode")
    axes[0, 1].set_ylabel("Reward")
    axes[0, 1].set_title("Step Reward Statistics")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Value function predictions
    axes[0, 2].plot(episodes, value_means, color='purple', linewidth=2)
    axes[0, 2].set_xlabel("Episode")
    axes[0, 2].set_ylabel("Mean Value")
    axes[0, 2].set_title("Value Function Predictions")
    axes[0, 2].grid(True, alpha=0.3)

    # Actor loss
    axes[1, 0].plot(episodes, actor_losses, color='red', linewidth=2)
    axes[1, 0].set_xlabel("Episode")
    axes[1, 0].set_ylabel("Actor Loss")
    axes[1, 0].set_title("Policy Loss Over Time")
    axes[1, 0].grid(True, alpha=0.3)

    # Clip fraction
    axes[1, 1].plot(episodes, clip_fracs, color='orange', linewidth=2)
    axes[1, 1].axhline(y=10, color='r', linestyle='--', alpha=0.5, label="10% threshold")
    axes[1, 1].set_xlabel("Episode")
    axes[1, 1].set_ylabel("Clip Fraction (%)")
    axes[1, 1].set_title("PPO Clipping Rate")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    # Learning rate (if tracked) or entropy coefficient
    entropy_coefs = [h.get("entropy_coef", 0) for h in history]
    axes[1, 2].plot(episodes, entropy_coefs, color='teal', linewidth=2)
    axes[1, 2].set_xlabel("Episode")
    axes[1, 2].set_ylabel("Entropy Coefficient")
    axes[1, 2].set_title("Entropy Coefficient Decay")
    axes[1, 2].set_yscale('log')
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "training_dynamics.png", dpi=150, bbox_inches='tight')
    plt.close()


def _generate_evolution_plots(history: List[Dict], output_dir: Path) -> None:
    """Generate neuroevolution-specific plots."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    # Filter evolution episodes
    evo_history = [h for h in history if "best_fitness" in h]
    if not evo_history:
        return

    episodes = [h["episode"] for h in evo_history]
    best_fitness = [h["best_fitness"] for h in evo_history]
    mean_fitness = [h["mean_fitness"] for h in evo_history]
    current_fitness = [h["current_fitness"] for h in evo_history]
    adopted = [h.get("evolution_adopted", 0) for h in evo_history]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Neuroevolution Progress", fontsize=16, fontweight='bold')

    # Fitness evolution
    axes[0].plot(episodes, best_fitness, label="Best", linewidth=2, marker='o')
    axes[0].plot(episodes, current_fitness, label="Current", linewidth=2, marker='s')
    axes[0].plot(episodes, mean_fitness, label="Population Mean", linewidth=2, marker='^')
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Fitness")
    axes[0].set_title("Evolution Fitness Over Time")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Adoption events
    adoption_episodes = [e for e, a in zip(episodes, adopted) if a > 0]
    adoption_fitness = [f for f, a in zip(best_fitness, adopted) if a > 0]
    axes[1].plot(episodes, best_fitness, alpha=0.5, label="Best Fitness")
    axes[1].scatter(adoption_episodes, adoption_fitness, color='red', s=100,
                   marker='*', label="Adopted", zorder=5)
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Fitness")
    axes[1].set_title("Evolution Adoption Events")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "evolution_progress.png", dpi=150, bbox_inches='tight')
    plt.close()


def _generate_reward_analysis_plots(history: List[Dict], output_dir: Path) -> None:
    """Generate reward component breakdown analysis."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    episodes = [h["episode"] for h in history]
    returns = [h["return"] for h in history]
    moving_avgs = [h.get("moving_avg", 0) for h in history]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Reward Analysis", fontsize=16, fontweight='bold')

    # Cumulative return growth
    cumulative_returns = np.cumsum(returns)
    axes[0, 0].plot(episodes, cumulative_returns, color='blue', linewidth=2)
    axes[0, 0].set_xlabel("Episode")
    axes[0, 0].set_ylabel("Cumulative Return")
    axes[0, 0].set_title("Cumulative Return Over Training")
    axes[0, 0].grid(True, alpha=0.3)

    # Return variance over time (rolling window)
    window = 10
    if len(returns) >= window:
        rolling_std = [np.std(returns[max(0, i-window):i+1]) for i in range(len(returns))]
        axes[0, 1].plot(episodes, rolling_std, color='orange', linewidth=2)
        axes[0, 1].set_xlabel("Episode")
        axes[0, 1].set_ylabel("Rolling Std Dev")
        axes[0, 1].set_title(f"Return Variance (window={window})")
        axes[0, 1].grid(True, alpha=0.3)

    # Return improvement rate
    if len(returns) > 1:
        improvement = np.diff(moving_avgs)
        axes[1, 0].plot(episodes[1:], improvement, color='green', linewidth=2)
        axes[1, 0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        axes[1, 0].set_xlabel("Episode")
        axes[1, 0].set_ylabel("Δ Moving Avg")
        axes[1, 0].set_title("Learning Progress (Moving Avg Improvement)")
        axes[1, 0].grid(True, alpha=0.3)

    # Return percentiles over time
    window = 10
    if len(returns) >= window:
        p25 = [np.percentile(returns[max(0, i-window):i+1], 25) for i in range(len(returns))]
        p50 = [np.percentile(returns[max(0, i-window):i+1], 50) for i in range(len(returns))]
        p75 = [np.percentile(returns[max(0, i-window):i+1], 75) for i in range(len(returns))]

        axes[1, 1].fill_between(episodes, p25, p75, alpha=0.3, color='blue', label='25-75%')
        axes[1, 1].plot(episodes, p50, color='blue', linewidth=2, label='Median')
        axes[1, 1].set_xlabel("Episode")
        axes[1, 1].set_ylabel("Return")
        axes[1, 1].set_title(f"Return Percentiles (window={window})")
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "reward_analysis.png", dpi=150, bbox_inches='tight')
    plt.close()


def _generate_learning_dynamics_plots(history: List[Dict], output_dir: Path) -> None:
    """Generate learning dynamics and stability analysis."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    episodes = [h["episode"] for h in history]
    value_losses = [h.get("value_loss", 0) for h in history]
    actor_losses = [h.get("actor_loss", 0) for h in history]
    entropies = [h.get("entropy", 0) for h in history]
    kls = [h.get("kl", 0) for h in history]
    returns = [h["return"] for h in history]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Learning Dynamics & Stability", fontsize=16, fontweight='bold')

    # Loss correlation
    axes[0, 0].scatter(value_losses, returns, alpha=0.5, s=20)
    axes[0, 0].set_xlabel("Value Loss")
    axes[0, 0].set_ylabel("Return")
    axes[0, 0].set_title("Value Loss vs Return")
    axes[0, 0].grid(True, alpha=0.3)

    # Actor vs Value loss
    axes[0, 1].scatter(actor_losses, value_losses, alpha=0.5, s=20, c=episodes, cmap='viridis')
    axes[0, 1].set_xlabel("Actor Loss")
    axes[0, 1].set_ylabel("Value Loss")
    axes[0, 1].set_title("Actor vs Value Loss (colored by episode)")
    axes[0, 1].grid(True, alpha=0.3)

    # KL vs Entropy
    axes[1, 0].scatter(kls, entropies, alpha=0.5, s=20, c=episodes, cmap='plasma')
    axes[1, 0].set_xlabel("KL Divergence")
    axes[1, 0].set_ylabel("Entropy")
    axes[1, 0].set_title("KL vs Entropy (colored by episode)")
    axes[1, 0].grid(True, alpha=0.3)

    # Training stability (rolling coefficient of variation)
    window = 10
    if len(returns) >= window:
        cv = [np.std(returns[max(0, i-window):i+1]) / (np.mean(returns[max(0, i-window):i+1]) + 1e-8)
              for i in range(len(returns))]
        axes[1, 1].plot(episodes, cv, color='red', linewidth=2)
        axes[1, 1].set_xlabel("Episode")
        axes[1, 1].set_ylabel("Coefficient of Variation")
        axes[1, 1].set_title(f"Training Stability (CV, window={window})")
        axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "learning_dynamics.png", dpi=150, bbox_inches='tight')
    plt.close()
