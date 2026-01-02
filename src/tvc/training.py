"""
Module: training
Purpose: Enhanced Recurrent PPO + Evolution training for 3D TVC.
Complexity: Time O(Epochs * Steps) | Space O(Buffer_Size)
Dependencies: jax, mujoco, optax
Last Updated: 2026-01-03
"""

from __future__ import annotations

import json
import logging
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple
import copy

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
from .policies import PolicyConfig, PolicyFunctions, build_policy_network, mutate_parameters

LOGGER = logging.getLogger(__name__)


@dataclass
class RunningNormalizer:
    """Welford's online normalization with stable updates."""
    count: float = 0.0
    mean: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=np.float32))
    m2: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=np.float32))

    @classmethod
    def initialise(cls, observation: np.ndarray) -> "RunningNormalizer":
        obs = np.asarray(observation, dtype=np.float32)
        return cls(count=1.0, mean=obs, m2=np.zeros_like(obs))

    def normalize(self, observation: np.ndarray, update_stats: bool = True) -> jnp.ndarray:
        obs = np.asarray(observation, dtype=np.float32)
        if self.mean.size == 0 or self.mean.shape != obs.shape:
            self.count = 1.0
            self.mean = obs.copy()
            self.m2 = np.zeros_like(obs)
            return jnp.asarray(obs, dtype=jnp.float32)

        if update_stats:
            self.count += 1.0
            delta = obs - self.mean
            self.mean = self.mean + delta / self.count
            delta2 = obs - self.mean
            self.m2 = self.m2 + delta * delta2
        
        variance = np.maximum(self.m2 / max(self.count, 1.0), 1e-4)
        std = np.sqrt(variance)
        normalized = (obs - self.mean) / (std + 1e-8)
        normalized = np.clip(normalized, -10.0, 10.0)
        return jnp.asarray(normalized, dtype=jnp.float32)


@dataclass
class TrainingState:
    """Recurrent PPO + Evolution training state."""
    params: Any
    opt_state: optax.OptState
    rng: jax.Array
    obs_rms: RunningNormalizer
    return_rms: RunningNormalizer = field(default_factory=lambda: RunningNormalizer()) # Track returns for normalization
    history: List[Dict[str, Any]] = field(default_factory=list)
    update_step: int = 0
    stage_index: int = 0
    stage_episode: int = 0
    best_return: float = -float("inf")
    elites: List[Any] = field(default_factory=list)
    moving_avg_return: float = 0.0
    moving_avg_alpha: float = 0.1
    total_successes: int = 0
    stage_successes: int = 0
    stage_attempts: int = 0
    recent_successes: List[bool] = field(default_factory=list)
    rolling_window_size: int = 30 # Increased for more stable curriculum transitions

    def update_rolling_success(self, success: bool) -> float:
        self.recent_successes.append(success)
        if len(self.recent_successes) > self.rolling_window_size:
            self.recent_successes.pop(0)
        return sum(self.recent_successes) / len(self.recent_successes) if self.recent_successes else 0.0


@dataclass(frozen=True)
class TrainingConfig:
    """Configuration for Recurrent PPO."""
    gamma: float = 0.99
    lam: float = 0.95
    learning_rate: float = 5e-4  # Slower for LSTM stability
    clip_epsilon: float = 0.2
    
    # Environment settings
    dt: float = 0.02
    max_episode_steps: int = 300
    eval_max_steps: int = 500
    
    # Sequence settings
    rollout_length: int = 3072  # Must be divisible by sequence_length (64*48=3072)
    sequence_length: int = 64  # Time horizon for LSTM backprop
    
    num_epochs: int = 4  # Reduced epochs for RNN safety
    minibatch_size: int = 32  # Number of sequences per batch
    
    value_clip_epsilon: float = 0.2
    grad_clip_norm: float = 0.5  # Tighter clipping for RNN
    entropy_coef: float = 0.05
    entropy_coef_decay: float = 0.998
    value_coef: float = 0.5
    weight_decay: float = 1e-5
    
    # Stability Constants (Refactored from hardcoded values)
    reward_clip_min: float = -100.0
    reward_clip_max: float = 100.0
    advantage_clip_min: float = -100.0
    advantage_clip_max: float = 100.0
    log_std_clip_min: float = -4.0
    log_std_clip_max: float = 2.0
    log_prob_clip_min: float = -100.0
    log_prob_clip_max: float = 100.0
    ratio_clip_limit: float = 10.0
    return_clip_limit: float = 1e4
    norm_return_clip: float = 10.0

    use_evolution: bool = True  
    population_size: int = 16
    elite_keep: int = 2
    mutation_scale: float = 0.05
    mutation_prob: float = 0.6
    evolution_interval: int = 10
    evolution_candidates: int = 4  # Number of mutants to evaluate per evolution step
    evolution_eval_episodes: int = 3  # Episodes to evaluate each candidate
    fitness_episodes: int = 3

    policy_config: PolicyConfig = PolicyConfig()
    rocket_params: RocketParams = RocketParams()


def _compute_gae(rewards, values, dones, gamma, lam):
    """Compute GAE with numerical stability."""
    # Clip rewards to prevent extreme advantage values (research-based fix)
    rewards = jnp.clip(rewards, config.reward_clip_min, config.reward_clip_max)
    
    def scan_fn(carry, inputs):
        reward, value, done = inputs
        next_value = carry
        delta = reward + gamma * next_value * (1.0 - done) - value
        advantage = delta + gamma * lam * (1.0 - done) * carry
        # Clip advantage during accumulation to prevent explosion
        # Increased clip range to allow more gradient signal
        advantage = jnp.clip(advantage, config.advantage_clip_min, config.advantage_clip_max)
        return advantage, advantage

    _, advantages_rev = jax.lax.scan(
        scan_fn,
        jnp.asarray(0.0, dtype=jnp.float32),
        (rewards, values[:-1], dones),
        reverse=True,
    )
    advantages = advantages_rev[::-1]
    returns = advantages + values[:-1] # Remove aggressive clipping, handle via normalization
    return advantages, returns


def _collect_rollout(
    env: TvcEnv,
    stage: CurriculumStage,
    state: TrainingState,
    funcs: PolicyFunctions,
    config: TrainingConfig,
    viewer: mujoco.viewer.Viewer | None = None,
) -> Tuple[Dict[str, jnp.ndarray], Dict[str, float]]:
    """Collect sequential rollout with non-blocking visualization."""
    
    # JIT-compiled helper for log probability with numerical stability
    @jax.jit
    def _gaussian_log_prob(mean, log_std, x):
        # Clip log_std to prevent numerical issues
        log_std = jnp.clip(log_std, config.log_std_clip_min, config.log_std_clip_max)
        var = jnp.exp(2 * log_std) + 1e-8  # Add epsilon for stability
        log_prob = -0.5 * jnp.sum(jnp.square(x - mean) / var + 2 * log_std + jnp.log(2 * np.pi), axis=-1)
        return jnp.clip(log_prob, -100.0, 100.0)  # Prevent extreme values
    
    env.configure_stage(stage)
    observation = env.reset()

    # Initialize LSTM hidden state (zero)
    # Shape: LSTMState tuple (h, c)
    # We use a batch size of 1 for rollout
    lstm_dim = config.policy_config.lstm_hidden_dim
    hidden = (jnp.zeros((1, lstm_dim)), jnp.zeros((1, lstm_dim)))
    
    # Batched buffers
    obs_buffer, action_buffer, logprob_buffer = [], [], []
    reward_buffer, value_buffer, done_buffer = [], [], []
    hidden_buffer_h, hidden_buffer_c = [], []

    obs_rms = state.obs_rms
    norm_observation = obs_rms.normalize(observation)

    episode_successes = []
    episode_returns = []
    current_episode_return = 0.0
    current_steps = 0
    
    # Non-blocking visualization timing
    last_render_time = time.time()
    target_render_fps = 60.0  # Target visual update rate (independent of sim speed)

    # FPS tracking for performance monitoring
    rollout_start_time = time.time()
    
    for step in range(config.rollout_length):
        state.rng, key = jax.random.split(state.rng)
        
        # Determine action (Single Step)
        # Input: [Batch=1, Features]
        # Hidden: [Batch=1, Hidden]
        # Dones: [Batch=1] (Always 0 during mid-step of rollout logic handled inside?)
        # Step: The RecurrentActorCritic resets state if done=1.
        # But here 'hidden' is carried.
        
        # Store PRE-UPDATE hidden state (to train on this step)
        hidden_buffer_h.append(hidden[0][0]) # Squeeze batch
        hidden_buffer_c.append(hidden[1][0])
        
        # Expand dims for batch=1
        obs_in = norm_observation[None, :] 
        dones_in = jnp.array([0.0]) # Always 0 for stepping, we handle reset manually
        
        # Policy Step
        # The actor_fn now returns (mean, new_hidden) or similar depending on sample
        mean, log_std, value, new_hidden = funcs.distribution(
            state.params, obs_in, hidden, dones_in, key=None, deterministic=True
        )
        
        # Clip std to prevent extreme noise (research-based stability fix)
        std = jnp.clip(jnp.exp(log_std), 1e-6, 1.0)
        epsilon = jax.random.normal(key, shape=mean.shape)
        action = mean + std * epsilon
        # Use proper action bounds: clip all to reasonable range
        action = jnp.clip(action, -1.0, 1.0)
        
        log_prob = _gaussian_log_prob(mean, log_std, action)
        
        # Squeeze batch
        action_squeezed = action[0]
        value_squeezed = value[0]
        log_prob_squeezed = log_prob[0]
        
        obs_buffer.append(norm_observation)
        action_buffer.append(action_squeezed)
        logprob_buffer.append(log_prob_squeezed)
        value_buffer.append(value_squeezed)
        
        # Env Step - ensure actions are valid (no NaN/Inf)
        action_np = np.asarray(action_squeezed, dtype=np.float32)
        # Replace NaN/Inf with safe defaults
        if not np.all(np.isfinite(action_np)):
            action_np = np.clip(np.nan_to_num(action_np, nan=0.0, posinf=0.1, neginf=-0.1), -0.14, 0.14)
        step_result = env.step(action_np)
        
        reward = step_result.reward
        done = step_result.done
        
        reward_buffer.append(reward)
        done_buffer.append(1.0 if done else 0.0)
        
        current_episode_return += reward
        current_steps += 1
        
        if viewer and viewer.is_running():
            # Non-blocking visualization: update at target FPS based on wall-clock time
            # This decouples training speed from visualization speed
            current_time = time.time()
            time_since_render = current_time - last_render_time
            
            # Render at target FPS (e.g., 60fps = every 0.0167s)
            if time_since_render >= 1.0 / target_render_fps:
                viewer.sync()
                last_render_time = current_time

        # Update loop state
        if done:
            # Check success
            # (Reuse existing success logic)
            pos_error = float(np.linalg.norm(env.data.qpos[0:3] - np.array(stage.target_position)))
            vel_error = float(np.linalg.norm(env.data.qvel[0:3] - np.array(stage.target_velocity)))
            # Simplified for brevity
            episode_success = pos_error < stage.position_tolerance and vel_error < stage.velocity_tolerance 
            
            episode_successes.append(episode_success)
            episode_returns.append(current_episode_return)
            current_episode_return = 0.0
            current_steps = 0
            
            # Reset Env
            observation = env.reset()
            norm_observation = obs_rms.normalize(observation)
            
            # Reset LSTM Hidden State for next step!
            hidden = (jnp.zeros_like(hidden[0]), jnp.zeros_like(hidden[1]))
        else:
            observation = step_result.observation
            norm_observation = obs_rms.normalize(observation)
            hidden = new_hidden # Propagate hidden state

    # Final bootstrap value
    obs_in = norm_observation[None, :]
    dones_in = jnp.array([0.0])
    _, _, final_value, _ = funcs.distribution(state.params, obs_in, hidden, dones_in, key=None)
    value_buffer.append(final_value[0])

    batch = {
        "observations": jnp.stack(obs_buffer),
        "actions": jnp.stack(action_buffer),
        "log_probs": jnp.stack(logprob_buffer),
        "rewards": jnp.array(reward_buffer, dtype=jnp.float32),
        "values": jnp.stack(value_buffer), # T+1
        "dones": jnp.array(done_buffer, dtype=jnp.float32),
        "hidden_h": jnp.stack(hidden_buffer_h),
        "hidden_c": jnp.stack(hidden_buffer_c),
    }

    # Calculate training throughput
    rollout_duration = time.time() - rollout_start_time
    steps_per_second = config.rollout_length / max(rollout_duration, 0.001)
    
    stats = {
        "episode_return": episode_returns[-1] if episode_returns else current_episode_return,
        "rollout_total_return": float(jnp.sum(batch["rewards"])),
        "episode_success": any(episode_successes) if episode_successes else False,
        "rollout_success_rate": sum(episode_successes)/len(episode_successes) if episode_successes else 0.0,
        "steps_per_second": steps_per_second,
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
    """Recurrent PPO Update."""
    
    # 1. GAE
    advantages, returns = _compute_gae(
        batch["rewards"], batch["values"], batch["dones"],
        config.gamma, config.lam
    )
    
    # Update return RMS and normalize returns for value function targets
    # This prevents the value loss from exploding when returns are in the thousands
    state.return_rms.normalize(np.asarray(returns), update_stats=True)
    norm_returns = (returns - state.return_rms.mean) / (np.sqrt(state.return_rms.m2 / max(state.return_rms.count, 1.0)) + 1e-8)
    norm_returns = jnp.clip(norm_returns, -config.norm_return_clip, config.norm_return_clip)

    # Normalize advantages
    advantages = (advantages - jnp.mean(advantages)) / (jnp.std(advantages) + 1e-8)
    
    # 2. Reshape into Sequences [Num_Seq, Seq_Len, Features]
    # Rollout length (e.g. 2048) must be divisible by sequence_length (e.g. 64)
    T = config.rollout_length
    S = config.sequence_length
    if T % S != 0:
        raise ValueError(f"Rollout length {T} not divisible by sequence length {S}")
    
    num_sequences = T // S
    
    def to_seq(x):
        # x is [T, ...] -> [Num_Seq, Seq_Len, ...]
        shape = (num_sequences, S) + x.shape[1:]
        return x[:num_sequences*S].reshape(shape)

    # Prepare dataset
    dataset = {
        "obs": to_seq(batch["observations"]),
        "actions": to_seq(batch["actions"]),
        "log_probs": to_seq(batch["log_probs"]),
        "advantages": to_seq(advantages),
        "returns": to_seq(norm_returns), # Use normalized returns!
        "values": to_seq(batch["values"][:-1]),
        "dones": to_seq(batch["dones"]),
        "hidden_h": to_seq(batch["hidden_h"])[:, 0, :], # Take only first hidden state of each sequence!
        "hidden_c": to_seq(batch["hidden_c"])[:, 0, :],
    }

    # Loss Function (Scanned over sequence) - with numerical stability
    def loss_fn(params, minibatch):
        # minibatch dims: [Batch, Seq_Len, Features]
        obs = minibatch["obs"]
        dones = minibatch["dones"]
        # Use initial hidden state from rollout
        init_h = minibatch["hidden_h"]
        init_c = minibatch["hidden_c"]
        hidden = (init_h, init_c)
        
        # Forward pass (RecurrentActorCritic handles scan)
        # mean, log_std, values: [Batch, Seq_Len, Features]
        mean, log_std, values, _ = funcs.distribution(
            params, obs, hidden, dones, deterministic=True
        )
        
        # Calc probabilities with numerical stability
        def log_prob_fn(m, l, a):
            # Clip log_std to prevent numerical issues
            l = jnp.clip(l, config.log_std_clip_min, config.log_std_clip_max)
            var = jnp.exp(2 * l) + 1e-8  # Add epsilon for stability
            log_prob = -0.5 * jnp.sum(jnp.square(a - m) / var + 2 * l + jnp.log(2 * np.pi), axis=-1)
            return jnp.clip(log_prob, config.log_prob_clip_min, config.log_prob_clip_max)  # Prevent extreme values
            
        new_log_probs = log_prob_fn(mean, log_std, minibatch["actions"])
        old_log_probs = jnp.clip(minibatch["log_probs"], config.log_prob_clip_min, config.log_prob_clip_max)
        
        # Standard PPO logic with clipped ratio for stability
        log_ratio = new_log_probs - old_log_probs
        log_ratio = jnp.clip(log_ratio, -config.ratio_clip_limit, config.ratio_clip_limit)  # Prevent exp explosion
        ratio = jnp.exp(log_ratio)
        clipped_ratio = jnp.clip(ratio, 1.0 - config.clip_epsilon, 1.0 + config.clip_epsilon)
        
        # Normalize advantages for stability
        advantages = minibatch["advantages"]
        advantages = (advantages - jnp.mean(advantages)) / (jnp.std(advantages) + 1e-8)
        
        surrogate = ratio * advantages
        clipped_surrogate = clipped_ratio * advantages
        actor_loss = -jnp.mean(jnp.minimum(surrogate, clipped_surrogate))
        
        # Clipped value function loss (PPO best practice for stability)
        old_values = minibatch["values"]
        returns = jnp.clip(minibatch["returns"], -config.return_clip_limit, config.return_clip_limit)  # Clip extreme returns
        value_clipped = old_values + jnp.clip(values - old_values, -config.value_clip_epsilon, config.value_clip_epsilon)
        value_loss_unclipped = jnp.square(returns - values)
        value_loss_clipped = jnp.square(returns - value_clipped)
        value_loss = 0.5 * jnp.mean(jnp.maximum(value_loss_unclipped, value_loss_clipped))
        
        # Entropy with stability
        log_std_clipped = jnp.clip(log_std, config.log_std_clip_min, config.log_std_clip_max)
        entropy = jnp.mean(jnp.sum(log_std_clipped + 0.5 * jnp.log(2.0 * jnp.pi * jnp.e), axis=-1))
        
        total_loss = actor_loss + config.value_coef * value_loss - current_entropy_coef * entropy
        
        # KL
        approx_kl = jnp.mean(old_log_probs - new_log_probs)
        
        return total_loss, (actor_loss, value_loss, entropy, approx_kl)

    # Optimization Loop
    params = state.params
    opt_state = state.opt_state
    
    for epoch in range(config.num_epochs):
        # Shuffle SEQUENCES
        inds = np.arange(num_sequences)
        np.random.shuffle(inds)
        
        for start in range(0, num_sequences, config.minibatch_size):
            end = start + config.minibatch_size
            idx = inds[start:end]
            
            mb = {k: v[idx] for k, v in dataset.items()}
            
            (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, mb)
            
            # Skip update if loss is NaN/Inf (gradient explosion protection)
            if not jnp.isfinite(loss):
                LOGGER.warning("Skipping update due to NaN/Inf loss")
                continue
            
            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            state.update_step += 1 # Increment update step!

    state.params = params
    state.opt_state = opt_state
    
    metrics = {
        "loss": float(loss),
        "actor_loss": float(aux[0]),
        "value_loss": float(aux[1]),
        "entropy": float(aux[2]),
        "kl": float(aux[3]),
    }
    return state, metrics


def _evaluate_candidate(env, stage, params, funcs, obs_rms, config, num_episodes=1):
    """Eval with LSTM - Production Grade."""
    total_r = 0.0
    lstm_dim = config.policy_config.lstm_hidden_dim
    
    for _ in range(num_episodes):
        env.configure_stage(stage)
        obs = env.reset()
        done = False
        
        # Initialize hidden state using config dimension
        hidden = (jnp.zeros((1, lstm_dim)), jnp.zeros((1, lstm_dim)))
        
        ep_r = 0
        steps = 0
        while not done and steps < config.eval_max_steps:
            norm_obs = obs_rms.normalize(obs, update_stats=False)
            
            mean, _, _, hidden = funcs.distribution(
                params, norm_obs[None, :], hidden, jnp.array([0.0]), deterministic=True
            )
            
            action = np.array(mean[0])
            # Protect against NaN/Inf actions
            if not np.all(np.isfinite(action)):
                action = np.clip(np.nan_to_num(action, nan=0.0, posinf=0.1, neginf=-0.1), -0.14, 0.14)
            
            res = env.step(action)
            ep_r += res.reward
            obs = res.observation
            done = res.done
            steps += 1
        total_r += ep_r
    return total_r / num_episodes


def train_controller(
    total_episodes: int,
    config: TrainingConfig = TrainingConfig(),
    output_dir: Path | None = None,
    seed: int = 42,
    resume_from: Path | None = None,
    visualize: bool = False,
) -> TrainingState:
    """Entry point for training."""
    env = TvcEnv(dt=config.dt, ctrl_limit=config.policy_config.action_limit, max_steps=config.max_episode_steps, seed=seed)
    
    # Sync physics
    real_params = config.rocket_params.from_model(env.model)
    import dataclasses
    config = dataclasses.replace(config, rocket_params=real_params)
    env.apply_rocket_params(config.rocket_params)
    
    curriculum = build_curriculum()
    
    # Init Policy (Recurrent)
    funcs = build_policy_network(config.policy_config)
    rng = jax.random.PRNGKey(seed)
    rng, init_key = jax.random.split(rng)
    
    sample_obs = env.reset()
    # Need sample hidden
    # Batch size 1
    sample_hidden = (jnp.zeros((1, config.policy_config.lstm_hidden_dim)), 
                    jnp.zeros((1, config.policy_config.lstm_hidden_dim)))
    
    params, _ = funcs.init(init_key, sample_obs[None, :], sample_hidden)
    
    obs_rms = RunningNormalizer.initialise(sample_obs)
    
    optimizer = optax.chain(
        optax.clip_by_global_norm(0.25),  # Tighter clipping for LSTM stability (research-based)
        optax.adamw(config.learning_rate, weight_decay=config.weight_decay, eps=1e-5)  # Explicit epsilon for stability
    )
    opt_state = optimizer.init(params)
    
    state = TrainingState(params, opt_state, rng, obs_rms, return_rms=RunningNormalizer())
    
    LOGGER.info("Recurrent PPO Training Initialized.")
    
    entropy_coef = config.entropy_coef
    
    viewer = None
    if visualize:
        viewer = mujoco.viewer.launch_passive(env.model, env.data)
        
        # ============================================================
        # VIEWER CONFIGURATION: Camera, Quality, Debug Overlays
        # ============================================================
        
        # Camera: Track the rocket body with good viewing angle
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = env.model.body('vehicle').id  # Track rocket (named 'vehicle' in XML)
        viewer.cam.distance = 25.0  # Zoom distance
        viewer.cam.azimuth = 135    # Viewing angle (degrees)
        viewer.cam.elevation = -15  # Slight upward look
        viewer.cam.lookat[:] = [0, 0, 10]  # Initial focus point
        
        # Visual Options: Minimal overlays (clean view)
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = False  # Hide contact forces
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = False  # Hide contact points
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_COM] = False           # Hide center of mass balls
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_ACTUATOR] = False      # Hide actuator activity
        
        # Rendering quality
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = False   # Solid objects
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_LIGHT] = True          # Enable lighting
        
        LOGGER.info("Viewer: Camera tracking rocket, debug overlays enabled")

    for episode in range(total_episodes):
        stage = curriculum[min(state.stage_index, len(curriculum)-1)]
        
        # Rollout
        batch, stats = _collect_rollout(env, stage, state, funcs, config, viewer)
        
        # Update
        state, metrics = _ppo_update(state, batch, optimizer, funcs, config, entropy_coef)
        
        # Update metrics
        state.moving_avg_return = (1.0 - state.moving_avg_alpha) * state.moving_avg_return + state.moving_avg_alpha * stats['episode_return']
        if stats['episode_return'] > state.best_return:
            state.best_return = float(stats['episode_return'])
        
        # Track successes
        if stats['episode_success']:
            state.total_successes += 1
            state.stage_successes += 1
        state.stage_attempts += 1
        
        rolling_sr = state.update_rolling_success(stats['episode_success'])
        
        if episode % 10 == 0:
            LOGGER.info(f"Ep {episode:4d} | R: {stats['episode_return']:8.1f} | Loss: {metrics['loss']:7.4f} | SR: {rolling_sr*100:4.1f}% | Stage: {state.stage_index}")
        
        # Advance Curriculum
        # If success rate > 80% and we've done at least min_episodes for the stage
        if rolling_sr >= 0.8 and state.stage_attempts >= stage.min_episodes:
            if state.stage_index < len(curriculum) - 1:
                state.stage_index += 1
                state.stage_attempts = 0 # Reset stage metrics
                state.stage_successes = 0
                state.recent_successes = [] # Clear window for new stage
                LOGGER.info(f"*** Advancing to Stage {state.stage_index}: {curriculum[state.stage_index].name} ***")
        
        # Decay entropy
        entropy_coef *= config.entropy_coef_decay
        
        # ============================================================
        # EVOLUTION: Safe Mutation through Gradients (SM-G) for LSTM
        # ============================================================
        if config.use_evolution and episode % config.evolution_interval == 0 and episode > 0:
            from .policies import safe_mutate_parameters_smg
            
            # Create sample data for sensitivity computation
            # sample_obs shape: [SeqLen, Features] -> [1, SeqLen, Features]
            sample_obs = batch["observations"][:config.sequence_length]
            sample_obs = sample_obs.reshape(1, sample_obs.shape[0], sample_obs.shape[1])  # Explicit reshape
            lstm_dim = config.policy_config.lstm_hidden_dim
            sample_hidden = (jnp.zeros((1, lstm_dim)), jnp.zeros((1, lstm_dim)))
            
            # Generate mutant candidates using SM-G
            best_params = state.params
            best_fitness = stats['episode_return']
            
            for candidate_idx in range(config.evolution_candidates):
                state.rng, mut_key = jax.random.split(state.rng)
                
                # Safe mutation - scales mutation inversely to output sensitivity
                mutant_params = safe_mutate_parameters_smg(
                    mut_key,
                    state.params,
                    funcs,
                    sample_obs,  # Already [1, SeqLen, Features]
                    sample_hidden,
                    scale=config.mutation_scale,
                )
                
                # Evaluate mutant
                mutant_fitness = _evaluate_candidate(
                    env, stage, mutant_params, funcs, state.obs_rms, config,
                    num_episodes=config.evolution_eval_episodes
                )
                
                if mutant_fitness > best_fitness:
                    best_fitness = mutant_fitness
                    best_params = mutant_params
                    LOGGER.info(f"  Evolution: Candidate {candidate_idx} improved! R: {mutant_fitness:.1f}")
            
            # Accept best if improved
            if best_params is not state.params:
                state.params = best_params
                LOGGER.info(f"  Evolution: Accepted mutant with R: {best_fitness:.1f}")
                
                # CRITICAL FIX: Reset optimizer state when params change discontinuously!
                # breakdown:
                # 1. The new params are a "jump" in the landscape.
                # 2. Old momentum (mu, nu) from Adam is now invalid/stale.
                # 3. Applying old momentum to new params causes massive instability.
                LOGGER.info("  Evolution: Resetting optimizer state to match new parameters.")
                state.opt_state = optimizer.init(state.params)
        
    return state



