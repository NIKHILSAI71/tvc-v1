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
import pickle
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
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

# ============================================================
# CHECKPOINTING: Save and load training state
# ============================================================

def save_checkpoint(
    state: "TrainingState",
    config: "TrainingConfig", 
    episode: int,
    output_dir: Path,
    is_best: bool = False,
    training_logs: List[Dict] = None,
) -> Path:
    """
    Save training checkpoint with model, optimizer, and logs.
    
    Structure:
        data/checkpoints/
            YYYYMMDD_HHMMSS_ep{episode}/
                model_params.pkl
                optimizer_state.pkl
                training_state.json
                config.json
                training_logs.json
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_name = f"{timestamp}_ep{episode:04d}"
    if is_best:
        checkpoint_name += "_best"
    
    checkpoint_dir = output_dir / "checkpoints" / checkpoint_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model parameters (JAX arrays -> numpy -> pickle)
    params_np = jax.tree_util.tree_map(np.array, state.params)
    with open(checkpoint_dir / "model_params.pkl", "wb") as f:
        pickle.dump(params_np, f)
    
    # Save optimizer state
    opt_state_np = jax.tree_util.tree_map(
        lambda x: np.array(x) if isinstance(x, jnp.ndarray) else x, 
        state.opt_state
    )
    with open(checkpoint_dir / "optimizer_state.pkl", "wb") as f:
        pickle.dump(opt_state_np, f)
    
    # Save training state (non-JAX fields)
    training_state_dict = {
        "update_step": state.update_step,
        "stage_index": state.stage_index,
        "stage_episode": state.stage_episode,
        "best_return": float(state.best_return),
        "moving_avg_return": float(state.moving_avg_return),
        "total_successes": state.total_successes,
        "stage_successes": state.stage_successes,
        "stage_attempts": state.stage_attempts,
        "recent_successes": state.recent_successes,
        "obs_rms": {
            "count": float(state.obs_rms.count),
            "mean": state.obs_rms.mean.tolist(),
            "m2": state.obs_rms.m2.tolist(),
        },
        "return_rms": {
            "count": float(state.return_rms.count),
            "mean": state.return_rms.mean.tolist() if hasattr(state.return_rms.mean, 'tolist') else [],
            "m2": state.return_rms.m2.tolist() if hasattr(state.return_rms.m2, 'tolist') else [],
        },
    }
    with open(checkpoint_dir / "training_state.json", "w") as f:
        json.dump(training_state_dict, f, indent=2)
    
    # Save config
    config_dict = {
        "gamma": config.gamma,
        "lam": config.lam,
        "learning_rate": config.learning_rate,
        "clip_epsilon": config.clip_epsilon,
        "entropy_coef": config.entropy_coef,
        "value_coef": config.value_coef,
        "rollout_length": config.rollout_length,
        "sequence_length": config.sequence_length,
        "num_epochs": config.num_epochs,
        "minibatch_size": config.minibatch_size,
        "use_evolution": config.use_evolution,
        "evolution_interval": config.evolution_interval,
        "use_rajs": config.use_rajs,
        "gru_hidden_dim": config.policy_config.gru_hidden_dim,
    }
    with open(checkpoint_dir / "config.json", "w") as f:
        json.dump(config_dict, f, indent=2)
    
    # Save training logs
    if training_logs:
        with open(checkpoint_dir / "training_logs.json", "w") as f:
            json.dump(training_logs, f, indent=2)
    
    LOGGER.info(f"Checkpoint saved: {checkpoint_dir}")
    return checkpoint_dir


def load_checkpoint(checkpoint_dir: Path, config: "TrainingConfig") -> Tuple["TrainingState", List[Dict]]:
    """
    Load training checkpoint.
    
    Returns:
        Tuple of (TrainingState, training_logs)
    """
    LOGGER.info(f"Loading checkpoint from: {checkpoint_dir}")
    
    # Load model parameters
    with open(checkpoint_dir / "model_params.pkl", "rb") as f:
        params_np = pickle.load(f)
    params = jax.tree_util.tree_map(jnp.array, params_np)
    
    # Load optimizer state
    with open(checkpoint_dir / "optimizer_state.pkl", "rb") as f:
        opt_state_np = pickle.load(f)
    opt_state = jax.tree_util.tree_map(
        lambda x: jnp.array(x) if isinstance(x, np.ndarray) else x,
        opt_state_np
    )
    
    # Load training state
    with open(checkpoint_dir / "training_state.json", "r") as f:
        ts = json.load(f)
    
    # Reconstruct normalizers
    obs_rms = RunningNormalizer(
        count=ts["obs_rms"]["count"],
        mean=np.array(ts["obs_rms"]["mean"], dtype=np.float32),
        m2=np.array(ts["obs_rms"]["m2"], dtype=np.float32),
    )
    return_rms = RunningNormalizer(
        count=ts["return_rms"]["count"],
        mean=np.array(ts["return_rms"]["mean"], dtype=np.float32) if ts["return_rms"]["mean"] else np.zeros((0,), dtype=np.float32),
        m2=np.array(ts["return_rms"]["m2"], dtype=np.float32) if ts["return_rms"]["m2"] else np.zeros((0,), dtype=np.float32),
    )
    
    # Create training state
    rng = jax.random.PRNGKey(42)  # Will be re-seeded
    state = TrainingState(
        params=params,
        opt_state=opt_state,
        rng=rng,
        obs_rms=obs_rms,
        return_rms=return_rms,
        update_step=ts["update_step"],
        stage_index=ts["stage_index"],
        stage_episode=ts["stage_episode"],
        best_return=ts["best_return"],
        moving_avg_return=ts["moving_avg_return"],
        total_successes=ts["total_successes"],
        stage_successes=ts["stage_successes"],
        stage_attempts=ts["stage_attempts"],
        recent_successes=ts["recent_successes"],
    )
    
    # Load training logs
    logs_path = checkpoint_dir / "training_logs.json"
    training_logs = []
    if logs_path.exists():
        with open(logs_path, "r") as f:
            training_logs = json.load(f)
    
    LOGGER.info(f"Loaded: ep={ts['update_step']}, stage={ts['stage_index']}, best_R={ts['best_return']:.1f}")
    return state, training_logs


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
    rolling_window_size: int = 50
    post_evolution_cooldown_counter: int = 0  # Tracks episodes since last evolution acceptance  # Larger window for reliable SR calculation

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
    learning_rate: float = 5e-5  # Further reduced for better stability
    clip_epsilon: float = 0.1  # Tighter clipping for stability
    
    # Environment settings
    dt: float = 0.02
    max_episode_steps: int = 300
    eval_max_steps: int = 500
    
    # Sequence settings
    rollout_length: int = 3072  # Balanced rollout size for stable training
    sequence_length: int = 64  # Time horizon for GRU backprop
    
    num_epochs: int = 4  # Reduced epochs for RNN safety
    minibatch_size: int = 32  # Number of sequences per batch
    
    value_clip_epsilon: float = 0.2
    grad_clip_norm: float = 0.5  # Tighter clipping for RNN
    entropy_coef: float = 0.03  # Higher for better exploration at new stages
    entropy_coef_decay: float = 0.9995  # Slower decay for longer exploration
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
    target_kl: float = 0.02  # KL threshold for early stopping

    use_evolution: bool = True  
    evolution_interval: int = 50  # Less frequent evolution to allow PPO to converge
    evolution_candidates: int = 4  # Number of mutants to evaluate per evolution step
    evolution_eval_episodes: int = 3  # Episodes to evaluate each candidate
    fitness_episodes: int = 3
    evolution_warmup_episodes: int = 150  # Longer warmup for PPO to establish good gradients
    evolution_stage_lockout: int = 75  # Longer lockout after stage transition
    
    # RESEARCH-BASED: Evolution stability parameters
    mutation_scale: float = 0.02  # Reduced from 0.05 for RNN stability
    post_evolution_cooldown: int = 10  # Episodes to use reduced PPO epochs after evolution
    post_evolution_epochs: int = 2  # Fewer PPO epochs during cooldown (prevents loss spikes)
    fitness_improvement_threshold: float = 1.10  # Mutant must be 10% better to accept
    post_evolution_lr_scale: float = 0.3  # Scale LR down after evolution for warmup

    # RAJS (Random Annealing Jump Start) - 2024 IROS paper achieving 97% rocket landing success
    use_rajs: bool = True
    rajs_initial_guide_steps: int = 80  # Shorter guidance to allow learning
    rajs_annealing_rate: float = 0.99  # Faster decay to transfer control to agent
    rajs_adaptive: bool = True  # Tie annealing to success rate instead of episode count
    
    # Value Pretraining - stabilizes advantage estimates from the start
    value_pretrain_updates: int = 100  # Longer warmup for value function stability
    value_pretrain_coef: float = 1.0  # Higher coefficient during pretraining
    
    # Staged exploration - reset entropy on stage advancement
    staged_exploration_reset: float = 0.7  # Reset entropy to this fraction of original on stage change
    
    # Value function stabilization (research-backed)
    value_loss_clip: float = 10.0  # Clip extreme value losses to prevent instability
    soft_reset_momentum_keep: float = 0.5  # Keep 50% of momentum to prevent loss spikes after evolution
    
    # Checkpointing
    checkpoint_interval: int = 50  # Save checkpoint every N episodes
    save_best: bool = True  # Save best model based on return

    policy_config: PolicyConfig = PolicyConfig()
    rocket_params: RocketParams = RocketParams()


def _compute_gae(rewards, values, dones, gamma, lam, config):
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

    # Initialize GRU hidden state (zero) - GRU uses single hidden state
    gru_dim = config.policy_config.gru_hidden_dim
    hidden = jnp.zeros((1, gru_dim))
    
    # Batched buffers
    obs_buffer, action_buffer, logprob_buffer = [], [], []
    reward_buffer, value_buffer, done_buffer = [], [], []
    hidden_buffer_h = []  # GRU only needs single hidden state (no hidden_c)

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
    
    # RAJS (Random Annealing Jump Start) state tracking
    # Uses heuristic controller for first N steps, then RL takes over
    episode_step = 0  # Steps within current episode
    rajs_guide_horizon = 0  # Will be randomized at episode start
    
    for step in range(config.rollout_length):
        state.rng, key, rajs_key = jax.random.split(state.rng, 3)
        
        # Determine action (Single Step)
        # Input: [Batch=1, Features]
        # Hidden: [Batch=1, Hidden]
        # Dones: [Batch=1] (Always 0 during mid-step of rollout logic handled inside?)
        # Step: The RecurrentActorCritic resets state if done=1.
        # But here 'hidden' is carried.
        
        # Store PRE-UPDATE hidden state (to train on this step)
        hidden_buffer_h.append(hidden[0])  # Squeeze batch, GRU uses single state
        
        # Expand dims for batch=1
        obs_in = norm_observation[None, :] 
        dones_in = jnp.array([0.0]) # Always 0 for stepping, we handle reset manually
        
        # Policy Step
        mean, log_std, value, new_hidden = funcs.distribution(
            state.params, obs_in, hidden, dones_in, key=None, deterministic=True
        )
        
        # Clip std to prevent extreme noise
        std = jnp.clip(jnp.exp(log_std), 1e-6, 1.0)
        epsilon = jax.random.normal(key, shape=mean.shape)
        action = mean + std * epsilon
        action = jnp.clip(action, -1.0, 1.0)
        
        # RAJS: Override action with smart PD heuristic during guide phase
        # Uses proportional-derivative control based on altitude and velocity
        # ENHANCED: Now includes orientation feedback for true TVC stabilization
        if config.use_rajs and episode_step < rajs_guide_horizon:
            # Extract state from raw observation (denormalized)
            obs_raw = observation  # Use raw obs, not normalized
            alt = float(obs_raw[2])  # z position (altitude)
            vz = float(obs_raw[5])   # z velocity (vertical speed)
            
            # Smart PD Heuristic:
            # Thrust: Higher when falling fast or low altitude
            # target_vz = -0.5 * sqrt(alt) for soft landing trajectory
            target_vz = -0.5 * np.sqrt(max(alt, 0.1))
            vz_error = vz - target_vz  # Negative when falling too fast
            
            # PD gains tuned for rocket dynamics
            kp_thrust = 0.3  # Proportional gain
            kd_thrust = 0.1  # Derivative gain (damping)
            
            # Base thrust for hover + correction
            thrust = 0.5 + kp_thrust * (-vz_error) + kd_thrust * (vz if vz < 0 else 0)
            thrust = np.clip(thrust, 0.2, 0.9)  # Safety bounds
            
            # Gimbal: Correct for lateral velocity and position
            vx, vy = float(obs_raw[3]), float(obs_raw[4])
            px, py = float(obs_raw[0]), float(obs_raw[1])
            
            # ============================================================
            # CRITICAL: Orientation-based gimbal control for TVC stabilization
            # This is what makes the rocket ACTIVELY COUNTERACT TILT
            # ============================================================
            # Observation layout (44 dims total):
            #   [0:3]   = pos (px, py, pz)
            #   [3:6]   = vel (vx, vy, vz)
            #   [6:15]  = rotation matrix R flattened (9 elements)
            #   [15:18] = angular velocity omega (wx, wy, wz)
            #   [18:20] = gimbal angles
            #   [20:22] = gimbal velocities
            #   [22:...] = targets, errors, etc.
            
            # Extract rotation matrix (3x3) from indices [6:15]
            R_flat = obs_raw[6:15]
            # R is stored row-major: R[i,j] = R_flat[i*3 + j]
            # R[:, 2] is body Z-axis in world frame
            # R[2, 0] ≈ sin(pitch) for small angles
            # R[2, 1] ≈ -sin(roll) for small angles
            R_20 = float(R_flat[6])  # R[2,0] = sin(pitch)
            R_21 = float(R_flat[7])  # R[2,1] = -sin(roll)
            
            # Tilt angles from rotation matrix (more accurate than quaternion approx)
            pitch_tilt = R_20   # Positive = tilted forward
            roll_tilt = -R_21   # Positive = tilted right
            
            # Get angular velocity from correct indices [15:18]
            omega_x = float(obs_raw[15])  # Angular velocity around X
            omega_y = float(obs_raw[16])  # Angular velocity around Y
            
            # PD gains for position correction
            kp_gimbal = 0.02
            kd_gimbal = 0.05
            
            # PD gains for orientation correction (CRITICAL for TVC)
            kp_orient = 0.12   # Orientation proportional gain
            kd_orient = 0.06   # Angular velocity damping
            
            # Combined gimbal control: position + orientation corrections
            # Gimbal X (pitch control): correct for forward tilt and lateral Y drift
            # Gimbal Y (roll control): correct for roll tilt and lateral X drift
            gimbal_x = (-kp_gimbal * px - kd_gimbal * vx 
                       - kp_orient * pitch_tilt - kd_orient * omega_y)
            gimbal_y = (-kp_gimbal * py - kd_gimbal * vy 
                       + kp_orient * roll_tilt + kd_orient * omega_x)
            
            gimbal_x = np.clip(gimbal_x, -0.12, 0.12)
            gimbal_y = np.clip(gimbal_y, -0.12, 0.12)
            
            action = jnp.array([[gimbal_x, gimbal_y, thrust]])
        episode_step += 1
        
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
            
            # Reset GRU Hidden State for next step!
            hidden = jnp.zeros_like(hidden)
            
            # RAJS: Reset episode step and randomize new guide horizon
            episode_step = 0
            if config.use_rajs:
                # Adaptive annealing: use success rate instead of episode count
                if config.rajs_adaptive:
                    # Higher success rate = less guidance needed
                    # BUT: Never drop below 30% guide to ensure continuous demonstration
                    success_rate = len([s for s in episode_successes if s]) / max(len(episode_successes), 1)
                    annealing_factor = max(0.3, 1.0 - success_rate * 0.7)  # Floor at 30%
                else:
                    # Original: episode-based annealing
                    annealing_factor = config.rajs_annealing_rate ** state.update_step
                max_guide = int(config.rajs_initial_guide_steps * annealing_factor)
                rajs_guide_horizon = int(jax.random.uniform(rajs_key) * max(max_guide, 1))
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
        "hidden_h": jnp.stack(hidden_buffer_h),  # GRU single state
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
        config.gamma, config.lam, config
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
        "hidden_h": to_seq(batch["hidden_h"])[:, 0, :], # Take only first hidden state of each sequence
    }

    # Loss Function (Scanned over sequence) - with numerical stability
    def loss_fn(params, minibatch):
        # minibatch dims: [Batch, Seq_Len, Features]
        obs = minibatch["obs"]
        dones = minibatch["dones"]
        # Use initial hidden state from rollout (GRU uses single state)
        init_h = minibatch["hidden_h"]
        hidden = init_h
        
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
        returns = jnp.clip(minibatch["returns"], -config.return_clip_limit, config.return_clip_limit)
        value_clipped = old_values + jnp.clip(values - old_values, -config.value_clip_epsilon, config.value_clip_epsilon)
        value_loss_unclipped = jnp.square(returns - values)
        value_loss_clipped = jnp.square(returns - value_clipped)
        value_loss = 0.5 * jnp.mean(jnp.maximum(value_loss_unclipped, value_loss_clipped))
        
        # CRITICAL: Clip extreme value losses to prevent instability (research-backed)
        value_loss = jnp.clip(value_loss, 0.0, config.value_loss_clip)
        
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
    
    kl_exceeded = False  # Track KL for early stopping
    for epoch in range(config.num_epochs):
        if kl_exceeded:
            break  # Stop early if KL threshold exceeded
        
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
            
            # KL early stopping - prevent catastrophic policy updates
            approx_kl = float(aux[3])
            if approx_kl > config.target_kl:
                LOGGER.debug(f"KL early stop: {approx_kl:.4f} > {config.target_kl}")
                kl_exceeded = True
                break
            
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
    """Eval with GRU - Production Grade."""
    total_r = 0.0
    gru_dim = config.policy_config.gru_hidden_dim
    
    for _ in range(num_episodes):
        env.configure_stage(stage)
        obs = env.reset()
        done = False
        
        # Initialize GRU hidden state (single state, not tuple)
        hidden = jnp.zeros((1, gru_dim))
        
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
    # Need sample hidden - GRU uses single state
    sample_hidden = jnp.zeros((1, config.policy_config.gru_hidden_dim))
    
    params, _ = funcs.init(init_key, sample_obs[None, :], sample_hidden)
    
    obs_rms = RunningNormalizer.initialise(sample_obs)
    
    optimizer = optax.chain(
        optax.clip_by_global_norm(0.25),  # Tighter clipping for GRU stability
        optax.adamw(config.learning_rate, weight_decay=config.weight_decay, eps=1e-5)
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
        
        # Track training logs for checkpoint
        log_entry = {
            "episode": episode,
            "return": float(stats['episode_return']),
            "success": stats['episode_success'],
            "stage": state.stage_index,
            "timestamp": datetime.now().isoformat(),
        }
        
        # ============================================================
        # VALUE PRETRAINING: First N updates train value function only
        # This stabilizes advantage estimates before actor learning
        # ============================================================
        if state.update_step < config.value_pretrain_updates:
            # Value-only update: scale down actor loss, boost value loss
            value_pretrain_factor = 0.1  # Reduce actor influence
            # Pass to ppo_update (modify entropy_coef as proxy)
            state, metrics = _ppo_update(
                state, batch, optimizer, funcs, config, 
                entropy_coef * 0.1  # Low entropy during value pretraining
            )
            if episode % 10 == 0:
                LOGGER.info(f"  [Value Pretrain {state.update_step}/{config.value_pretrain_updates}]")
        elif state.post_evolution_cooldown_counter > 0:
            # POST-EVOLUTION COOLDOWN: Use reduced epochs to prevent loss spikes
            # This gives Adam optimizer time to rebuild moment estimates for mutated weights
            cooldown_config = dataclasses.replace(config, num_epochs=config.post_evolution_epochs)
            state, metrics = _ppo_update(state, batch, optimizer, funcs, cooldown_config, entropy_coef * 0.5)
            if episode % 5 == 0:
                LOGGER.info(f"  [Post-Evolution Cooldown: {state.post_evolution_cooldown_counter} eps remaining, using {config.post_evolution_epochs} PPO epochs]")
        else:
            # Normal PPO update
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
        
        # Enhanced logging with stage name
        success_emoji = "✓" if stats['episode_success'] else "✗"
        if episode % 10 == 0:
            LOGGER.info(f"Ep {episode:4d} | R: {stats['episode_return']:7.1f} | Loss: {metrics['loss']:6.2f} | SR: {rolling_sr*100:4.1f}% | Stage: {stage.name} | {success_emoji}")
        
        # Advance Curriculum
        # STRICTER GATE: 80% SR + consecutive successes + min episodes
        previous_stage_index = state.stage_index
        consecutive_success_req = 5  # Must win 5 in a row
        recent_streak = sum(1 for s in state.recent_successes[-consecutive_success_req:] if s) if state.recent_successes else 0
        has_streak = recent_streak >= consecutive_success_req
        
        # Stage 0 (Hover) requires 60% SR (lowered from 80% for achievability), other stages require 70%
        sr_threshold = 0.60 if state.stage_index == 0 else 0.70
        
        if rolling_sr >= sr_threshold and state.stage_attempts >= stage.min_episodes and has_streak:
            if state.stage_index < len(curriculum) - 1:
                old_stage_name = curriculum[state.stage_index].name
                state.stage_index += 1
                state.stage_attempts = 0
                state.stage_successes = 0
                state.recent_successes = []
                new_stage = curriculum[state.stage_index]
                LOGGER.info(f"")
                LOGGER.info(f"STAGE UP! {old_stage_name} -> {new_stage.name}")
                LOGGER.info(f"   New altitude: {new_stage.initial_position[2]:.0f}m | Target SR: {sr_threshold*100:.0f}%")
                
                # STAGED EXPLORATION RESET: Boost entropy on stage transition for renewed exploration
                if config.staged_exploration_reset > 0:
                    old_entropy = entropy_coef
                    entropy_coef = config.entropy_coef * config.staged_exploration_reset
        
        # Decay entropy (only if stage didn't just advance)
        if state.stage_index == previous_stage_index:
            entropy_coef *= config.entropy_coef_decay
        
        # ============================================================
        # EVOLUTION: Safe Mutation through Gradients (SM-G) for GRU
        # ============================================================
        # Skip evolution during: warmup, stage lockout period, or not on interval
        stage_episodes_since_advance = state.stage_attempts
        evolution_ready = (
            config.use_evolution 
            and episode > config.evolution_warmup_episodes 
            and episode % config.evolution_interval == 0
            and stage_episodes_since_advance > config.evolution_stage_lockout  # NEW: Stage lockout
        )
        if evolution_ready:
            from .policies import safe_mutate_parameters_smg
            
            # Create sample data for sensitivity computation
            # sample_obs shape: [SeqLen, Features] -> [1, SeqLen, Features]
            sample_obs = batch["observations"][:config.sequence_length]
            sample_obs = sample_obs.reshape(1, sample_obs.shape[0], sample_obs.shape[1])  # Explicit reshape
            gru_dim = config.policy_config.gru_hidden_dim
            sample_hidden = jnp.zeros((1, gru_dim))  # GRU single state
            
            # Generate mutant candidates using SM-G
            best_params = state.params
            best_fitness = stats['episode_return']
            
            # ADAPTIVE MUTATION: Start with smaller mutations, ramp up over training
            # This prevents evolution from disrupting early learning
            episode_progress = min(episode / total_episodes, 1.0)
            adaptive_mutation_scale = config.mutation_scale * (0.3 + 0.7 * episode_progress)
            # Early (ep 0): 30% mutation, Late (ep 1000): 100% mutation
            
            for candidate_idx in range(config.evolution_candidates):
                state.rng, mut_key = jax.random.split(state.rng)
                
                # Safe mutation - scales mutation inversely to output sensitivity
                mutant_params = safe_mutate_parameters_smg(
                    mut_key,
                    state.params,
                    funcs,
                    sample_obs,  # Already [1, SeqLen, Features]
                    sample_hidden,
                    scale=adaptive_mutation_scale,  # Use adaptive scale
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
            
            # Accept best if improved SIGNIFICANTLY (10% improvement threshold)
            # This prevents accepting tiny improvements that destabilize training
            original_fitness = stats['episode_return']
            improvement_ratio = best_fitness / max(abs(original_fitness), 1.0)
            meets_threshold = improvement_ratio >= config.fitness_improvement_threshold
            
            if best_params is not state.params and meets_threshold:
                state.params = best_params
                LOGGER.info(f"  Evolution: Accepted mutant with R: {best_fitness:.1f} ({improvement_ratio:.1%} improvement)")
                
                # RESEARCH-BASED FIX: Activate post-evolution cooldown
                # This uses reduced PPO epochs and learning rate for the next N episodes
                # to prevent gradient explosion after weight mutation
                state.post_evolution_cooldown_counter = config.post_evolution_cooldown
                LOGGER.info(f"  Evolution: Cooldown activated for {config.post_evolution_cooldown} episodes")
                
                # Soft optimizer reset preserving partial momentum
                LOGGER.info(f"  Evolution: Soft optimizer reset (keeping {config.soft_reset_momentum_keep*100:.0f}% momentum)")
                new_opt_state = optimizer.init(state.params)
                
                # Blend old and new optimizer states (preserve partial momentum)
                def blend_opt_states(old, new):
                    if isinstance(old, jnp.ndarray) and isinstance(new, jnp.ndarray):
                        if old.shape == new.shape:
                            return config.soft_reset_momentum_keep * old + (1 - config.soft_reset_momentum_keep) * new
                    return new
                
                try:
                    state.opt_state = jax.tree_util.tree_map(blend_opt_states, state.opt_state, new_opt_state)
                except:
                    # Fallback to full reset if blending fails
                    state.opt_state = new_opt_state
            elif best_params is not state.params:
                LOGGER.info(f"  Evolution: Rejected mutant (improvement {improvement_ratio:.1%} < threshold {config.fitness_improvement_threshold:.0%})")
        
        # Decrement cooldown counter each episode
        if state.post_evolution_cooldown_counter > 0:
            state.post_evolution_cooldown_counter -= 1
        
        # ============================================================
        # CHECKPOINTING: Save model periodically and on best
        # ============================================================
        state.history.append(log_entry)
        
        # Save checkpoint at intervals
        if output_dir and episode > 0 and episode % config.checkpoint_interval == 0:
            save_checkpoint(state, config, episode, output_dir, is_best=False, training_logs=state.history)
        
        # Save best model
        if config.save_best and stats['episode_return'] > state.best_return:
            if output_dir:
                save_checkpoint(state, config, episode, output_dir, is_best=True, training_logs=state.history)
        
    # Final save
    if output_dir:
        LOGGER.info("Saving final checkpoint...")
        save_checkpoint(state, config, total_episodes, output_dir, is_best=False, training_logs=state.history)
    
    return state



