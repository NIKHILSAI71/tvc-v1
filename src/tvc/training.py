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


@dataclass(frozen=True)
class TrainingConfig:
    """Enhanced PPO + Neuroevolution configuration.

    Based on successful rocket stabilization system:
    - Large population (20+ candidates)
    - Score-based selection with top performers
    - Mutation of best candidates to create next generation
    - Combined with PPO gradient-based learning
    """
    # PPO settings - Tuned for continuous control
    gamma: float = 0.99
    lam: float = 0.95
    learning_rate: float = 1e-4  # Reduced for stability
    clip_epsilon: float = 0.15  # Tighter clipping for smoother learning
    rollout_length: int = 512  # Longer rollouts for better value estimation
    num_epochs: int = 8  # Reduced to prevent overf itting
    minibatch_size: int = 128  # Larger batches for stable gradients
    value_clip_epsilon: float = 0.15
    grad_clip_norm: float = 1.0  # Increased for stability
    entropy_coef: float = 0.02  # Higher initial exploration
    entropy_coef_decay: float = 0.998  # Slower decay
    value_coef: float = 1.0  # Increased value function importance
    weight_decay: float = 0.0  # Disabled - let evolution handle regularization

    # Neuroevolution settings - Based on rocket stabilization paper
    use_evolution: bool = True
    population_size: int = 20  # Large population like the 1000-network system (scaled down)
    elite_keep: int = 5  # Keep top 5 performers
    mutation_scale: float = 0.05  # Larger mutations for exploration
    mutation_prob: float = 0.8  # 80% of parameters mutated
    evolution_interval: int = 3  # More frequent evolution
    fitness_episodes: int = 3  # Evaluate each candidate thoroughly

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
) -> Tuple[Dict[str, jnp.ndarray], Dict[str, float]]:
    """Collect rollout trajectory."""
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

        reward_buffer.append(step_result.reward)
        done_buffer.append(1.0 if step_result.done else 0.0)

        if step_result.done:
            observation = env.reset()
            norm_observation = obs_rms.normalize(observation)
        else:
            norm_observation = obs_rms.normalize(step_result.observation)

    # Final value for bootstrap
    _, _, final_value = funcs.distribution(
        state.params,
        norm_observation,
        key=None,
        deterministic=True,
    )
    value_buffer.append(final_value)

    batch = {
        "observations": jnp.stack(obs_buffer),
        "actions": jnp.stack(action_buffer),
        "log_probs": jnp.stack(logprob_buffer),
        "rewards": jnp.array(reward_buffer, dtype=jnp.float32),
        "values": jnp.stack(value_buffer),
        "dones": jnp.array(done_buffer, dtype=jnp.float32),
    }

    stats = {
        "episode_return": float(jnp.sum(batch["rewards"])),
        "reward_mean": float(jnp.mean(batch["rewards"])),
        "reward_std": float(jnp.std(batch["rewards"])),
        "value_mean": float(jnp.mean(batch["values"][:-1])),
        "action_mean": float(jnp.mean(jnp.abs(batch["actions"]))),
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

        approx_kl = jnp.mean(minibatch["log_probs"] - log_prob)
        approx_kl = jnp.nan_to_num(approx_kl, nan=0.0)

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

    # Multiple epochs over the dataset
    for epoch in range(config.num_epochs):
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
) -> TrainingState:
    """Enhanced PPO + Evolution training with visualization."""

    LOGGER.info("=" * 80)
    LOGGER.info("🚀 ENHANCED TVC TRAINING - PPO + Evolution")
    LOGGER.info("=" * 80)
    LOGGER.info("Episodes: %d | Rollout: %d | Evolution: %s",
                total_episodes, config.rollout_length, config.use_evolution)
    LOGGER.info("=" * 80)

    # Create environment
    env = TvcEnv(dt=0.02, ctrl_limit=0.3, max_steps=2000, seed=seed)
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

    # Setup optimizer
    optimizer = optax.chain(
        optax.clip_by_global_norm(config.grad_clip_norm),
        optax.adamw(
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
        ),
    )
    opt_state = optimizer.init(params)

    state = TrainingState(
        params=params,
        opt_state=opt_state,
        rng=rng,
        obs_rms=obs_rms,
    )

    LOGGER.info("✅ Training initialized | Policy params: %d",
                sum(p.size for p in jax.tree_util.tree_leaves(params)))

    current_entropy_coef = config.entropy_coef
    start_time = time.time()

    # Training loop
    for episode in range(total_episodes):
        episode_start = time.time()

        # Select curriculum stage
        stage_idx = min(state.stage_index, len(curriculum) - 1)
        stage = curriculum[stage_idx]

        # Collect rollout
        batch, rollout_stats = _collect_rollout(env, stage, state, policy_funcs, config)

        # PPO update
        state, update_metrics = _ppo_update(state, batch, optimizer, policy_funcs, config, current_entropy_coef)

        # Decay entropy coefficient
        current_entropy_coef *= config.entropy_coef_decay
        current_entropy_coef = max(current_entropy_coef, 1e-4)

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
        state.best_return = max(state.best_return, episode_return)
        state.moving_avg_return = (state.moving_avg_alpha * episode_return +
                                   (1 - state.moving_avg_alpha) * state.moving_avg_return)

        episode_time = time.time() - episode_start
        elapsed = time.time() - start_time

        metrics = {
            "episode": episode + 1,
            "stage": stage.name,
            "return": episode_return,
            "best_return": state.best_return,
            "moving_avg": state.moving_avg_return,
            "episode_time": episode_time,
            **rollout_stats,
            **update_metrics,
            **evolution_metrics,
        }

        state.history.append(metrics)

        # Enhanced logging
        if (episode + 1) % 1 == 0:  # Log every episode
            LOGGER.info(
                "Ep %3d/%d | Stage: %-20s | Ret: %7.2f | Avg: %7.2f | Best: %7.2f | "
                "Loss: %6.3f | VLoss: %6.3f | Ent: %.3f | KL: %.4f | Clip: %.2f%% | Time: %.1fs",
                episode + 1,
                total_episodes,
                stage.name,
                episode_return,
                state.moving_avg_return,
                state.best_return,
                update_metrics["loss"],
                update_metrics["value_loss"],
                update_metrics["entropy"],
                update_metrics["kl"],
                update_metrics["clip_frac"] * 100,
                episode_time,
            )

        # Stage progression
        state.stage_episode += 1
        if state.stage_episode >= stage.episodes and state.stage_index < len(curriculum) - 1:
            state.stage_index += 1
            state.stage_episode = 0
            LOGGER.info("=" * 80)
            LOGGER.info("📈 STAGE PROGRESSION: %s → %s",
                       stage.name, curriculum[state.stage_index].name)
            LOGGER.info("=" * 80)

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
    LOGGER.info("=" * 80)
    LOGGER.info("🎉 TRAINING COMPLETE")
    LOGGER.info("Total time: %.1f min | Best return: %.2f | Final avg: %.2f",
                total_time / 60, state.best_return, state.moving_avg_return)
    LOGGER.info("=" * 80)

    return state


def _generate_plots(history: List[Dict], output_dir: Path) -> None:
    """Generate training visualization plots."""
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

    # Value Loss
    axes[0, 1].plot(episodes, value_losses, color='orange')
    axes[0, 1].set_xlabel("Episode")
    axes[0, 1].set_ylabel("Value Loss")
    axes[0, 1].set_title("Value Function Loss")
    axes[0, 1].grid(True, alpha=0.3)

    # Entropy
    axes[0, 2].plot(episodes, entropies, color='green')
    axes[0, 2].set_xlabel("Episode")
    axes[0, 2].set_ylabel("Entropy")
    axes[0, 2].set_title("Policy Entropy")
    axes[0, 2].grid(True, alpha=0.3)

    # KL Divergence
    axes[1, 0].plot(episodes, kls, color='purple')
    axes[1, 0].axhline(y=0.02, color='r', linestyle='--', alpha=0.5, label="Target")
    axes[1, 0].set_xlabel("Episode")
    axes[1, 0].set_ylabel("KL Divergence")
    axes[1, 0].set_title("KL Divergence")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Return Distribution
    axes[1, 1].hist(returns, bins=20, alpha=0.7, color='blue', edgecolor='black')
    axes[1, 1].axvline(x=np.mean(returns), color='r', linestyle='--', linewidth=2, label=f"Mean: {np.mean(returns):.2f}")
    axes[1, 1].set_xlabel("Return")
    axes[1, 1].set_ylabel("Frequency")
    axes[1, 1].set_title("Return Distribution")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    # Training Summary
    axes[1, 2].axis('off')
    summary_text = f"""
    Training Summary
    ================
    Total Episodes: {len(history)}

    Returns:
      Best: {max(returns):.2f}
      Mean: {np.mean(returns):.2f}
      Final Avg: {moving_avgs[-1]:.2f}

    Final Metrics:
      Value Loss: {value_losses[-1]:.4f}
      Entropy: {entropies[-1]:.3f}
      KL: {kls[-1]:.4f}
    """
    axes[1, 2].text(0.1, 0.5, summary_text, fontsize=12, family='monospace',
                    verticalalignment='center')

    plt.tight_layout()
    plot_path = output_dir / "training_progress.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    LOGGER.info("📊 Plot saved: %s", plot_path)