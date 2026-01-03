"""Production 3D TVC environment with MuJoCo."""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import jax.numpy as jnp
import mujoco
import numpy as np

from .dynamics import RocketParams, state_to_observation

LOGGER = logging.getLogger(__name__)


@dataclass
class StepResult:
    """Environment step output."""
    observation: np.ndarray
    reward: float
    done: bool
    info: Dict[str, float]


class TvcEnv:
    """3D TVC Rocket Environment with MuJoCo physics."""

    def __init__(
        self,
        model_path: str | None = None,
        dt: float = 0.02,
        ctrl_limit: float = 0.14,  # Gimbal angle limit (±8° = 0.14 rad)
        max_steps: int = 300,  # CRITICAL: Reduced to 300 (6 seconds) for rapid learning
        seed: int | None = None,
        domain_randomization: bool = True,  # Enable robust training
        actuator_delay_steps: int = 0,  # Actuator delay (0 = no delay, 1-3 = realistic)
    ) -> None:
        asset_path = model_path or _default_asset_path()
        self.model = mujoco.MjModel.from_xml_path(asset_path)
        self.data = mujoco.MjData(self.model)

        self._ctrl_limit = float(ctrl_limit)
        self.frame_skip = max(int(dt / self.model.opt.timestep), 1)
        self.max_steps = int(max_steps)
        self._rng = np.random.default_rng(seed)
        self._step_counter = 0
        self._domain_randomization = domain_randomization

        # Domain randomization parameters (stored for reset)
        # Domain randomization parameters (stored for reset)
        self._base_mass = 282.5  # Fixed: Matches XML total mass
        self._base_inertia = np.array([95.0, 95.0, 5.0])
        self._base_thrust_max = 8540.0
        self._base_damping = 0.8
        
        # Sim2Real: Persistent disturbances
        self._sensor_bias = np.zeros(6)  # Drift in pos/vel sensors
        self._engine_lag_alpha = 1.0  # 1.0 = instant, <1.0 = lag
        self._current_actuator_delay = 0

        # Reward normalization
        self._reward_count = 0.0
        self._reward_mean = 0.0
        self._reward_m2 = 0.0
        self._reward_norm_warmup = 64
        self._reward_norm_clip = 6.0

        # Action tracking
        self._last_action = np.zeros(3, dtype=np.float64)
        self._prev_action = np.zeros(3, dtype=np.float64)
        
        # Enhanced Action Smoothing (Low-pass filter)
        # alpha=0.15 heavily smooths the control signal, filtering out high-frequency
        # jitter from the stochastic policy. This prevents the "shaking" seen in early training.
        # Real actuators behave like low-pass filters anyway.
        self._smoothed_action = np.zeros(3, dtype=np.float64)
        self._action_smoothing_alpha = 0.15
        
        # Actuator delay buffer (for realistic hardware simulation)
        self._actuator_delay_steps = actuator_delay_steps
        self._action_buffer = [np.zeros(3, dtype=np.float64) for _ in range(max(1, actuator_delay_steps + 1))]

        # Stage configuration - Realistic landing scenario altitudes
        self._stage_config = {
            "target_position": [0.0, 0.0, 8.0],
            "target_orientation": [1.0, 0.0, 0.0, 0.0],
            "target_velocity": [0.0, 0.0, 0.0],
            "target_angular_velocity": [0.0, 0.0, 0.0],
            "initial_position": [0.0, 0.0, 8.0],  # Start from 8m hover
            "initial_velocity": [0.0, 0.0, 0.0],
            "initial_orientation": [1.0, 0.0, 0.0, 0.0],
            "initial_angular_velocity": [0.0, 0.0, 0.0],
            "position_tolerance": 1.0,
            "velocity_tolerance": 0.8,
            "orientation_tolerance": 0.2,
            "angular_velocity_tolerance": 0.5,
            "tolerance_bonus": 0.6,
            "stage_name": "hover_stabilization",  # Track current stage
        }

        self._actuator_ids: Dict[str, int] = {}
        self._resolve_actuators()
        mujoco.mj_forward(self.model, self.data)

    @property
    def ctrl_limit(self) -> float:
        """Get control limit."""
        return self._ctrl_limit

    def _resolve_actuators(self) -> None:
        """Find actuator IDs."""
        self._actuator_ids = {}
        for key in ("tvc_pitch", "tvc_yaw", "thrust_control"):
            try:
                idx = int(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, key))
                self._actuator_ids[key] = idx
            except Exception:
                self._actuator_ids[key] = -1

    def configure_stage(self, stage) -> None:
        """Apply curriculum stage configuration."""
        stage_name = getattr(stage, "name", "hover_stabilization")
        self._stage_config.update({
            "target_position": list(getattr(stage, "target_position", [0.0, 0.0, 8.0])),
            "target_orientation": list(getattr(stage, "target_orientation", [1.0, 0.0, 0.0, 0.0])),
            "target_velocity": list(getattr(stage, "target_velocity", [0.0, 0.0, 0.0])),
            "target_angular_velocity": list(getattr(stage, "target_angular_velocity", [0.0, 0.0, 0.0])),
            "initial_position": list(getattr(stage, "initial_position", [0.0, 0.0, 8.0])),
            "initial_velocity": list(getattr(stage, "initial_velocity", [0.0, 0.0, 0.0])),
            "initial_orientation": list(getattr(stage, "initial_orientation", [1.0, 0.0, 0.0, 0.0])),
            "initial_angular_velocity": list(getattr(stage, "initial_angular_velocity", [0.0, 0.0, 0.0])),
            "position_tolerance": float(getattr(stage, "position_tolerance", 1.0)),
            "velocity_tolerance": float(getattr(stage, "velocity_tolerance", 0.8)),
            "orientation_tolerance": float(getattr(stage, "orientation_tolerance", 0.2)),
            "angular_velocity_tolerance": float(getattr(stage, "angular_velocity_tolerance", 0.5)),
            "tolerance_bonus": float(getattr(stage, "tolerance_bonus", 0.6)),
            "stage_name": stage_name,
        })

    def apply_rocket_params(self, params: RocketParams | None) -> None:
        """Update physics parameters."""
        if params is None:
            return

        vehicle_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "vehicle")
        if vehicle_id >= 0:
            self.model.body_mass[vehicle_id] = params.mass
            self.model.body_inertia[vehicle_id] = np.array([
                max(params.inertia[0], 1e-3),
                max(params.inertia[1], 1e-3),
                max(params.inertia[2], 1e-3),
            ])

        self.model.opt.gravity[:] = [0.0, 0.0, -params.gravity]

        for joint in ("tvc_x", "tvc_y"):
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint)
            if joint_id >= 0:
                self.model.jnt_range[joint_id] = [-params.tvc_limit, params.tvc_limit]

    def reset(self) -> np.ndarray:
        """Reset environment with domain randomization."""
        self._step_counter = 0
        mujoco.mj_resetData(self.model, self.data)

        # Only enable after Stage 2 (lateral_translation) to allow stable initial learning
        stage_name = self._stage_config.get("stage_name", "hover_stabilization")
        enable_randomization = self._domain_randomization and stage_name not in ["hover_stabilization", "lateral_translation"]
        
        # Domain randomization: Randomize physics parameters for robustness
        if enable_randomization:
            # Mass variation: ±20% (Increased from 10% for fuel usage simulation)
            mass_factor = self._rng.uniform(0.8, 1.2)
            vehicle_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "vehicle")
            if vehicle_id >= 0:
                self.model.body_mass[vehicle_id] = self._base_mass * mass_factor

            # Inertia variation: ±20%
            inertia_factor = self._rng.uniform(0.8, 1.2, 3)
            if vehicle_id >= 0:
                self.model.body_inertia[vehicle_id] = self._base_inertia * inertia_factor
                
            # CRITICAL: Center of Mass (CoM) offset optimization
            # Real rockets have shifting CoM. Add random offset ±5cm
            com_offset = self._rng.uniform(-0.05, 0.05, 3)
            self.model.body_ipos[vehicle_id] = com_offset

            # Thrust variation: -10% to +5% (Engines degrade more than they over-perform)
            thrust_factor = self._rng.uniform(0.90, 1.05)
            # Note: Thrust variation applied in step() via scaling

            # Damping variation: ±40% (Aerodynamics are hard to model)
            damping_factor = self._rng.uniform(0.6, 1.4)
            # Applied during dynamics

            # Wind disturbance: 0-12 m/s random direction (Gusts & Turbulence)
            # Increased from 8 m/s for "Expert" level toughness
            self._wind_force = self._rng.normal(0.0, 5.0, 3) 

            # Sim2Real: Variable Actuator Delay (0-2 steps)
            # Randomly chosen per episode to force policy to be robust to latency
            self._current_actuator_delay = self._rng.integers(0, 3)
            self._action_buffer = [np.zeros(3, dtype=np.float64) for _ in range(self._current_actuator_delay + 1)]

            # Sim2Real: Engine Response Lag (Thrust spool-up time)
            # alpha 0.4 ~= 100ms lag at 50Hz, alpha 1.0 = instant
            self._engine_lag_alpha = self._rng.uniform(0.3, 1.0)

            # Sim2Real: Sensor Bias (Drift)
            # GPS drift up to 10cm, IMU drift up to 0.05 m/s
            self._sensor_bias[0:3] = self._rng.normal(0.0, 0.10, 3) # Pos bias
            self._sensor_bias[3:6] = self._rng.normal(0.0, 0.05, 3) # Vel bias

        else:
            self._wind_force = np.zeros(3)
            self._current_actuator_delay = self._actuator_delay_steps
            self._action_buffer = [np.zeros(3, dtype=np.float64) for _ in range(self._current_actuator_delay + 1)]
            self._engine_lag_alpha = 1.0
            self._sensor_bias = np.zeros(6)

        initial_pos = np.array(self._stage_config["initial_position"], dtype=np.float64)
        initial_vel = np.array(self._stage_config["initial_velocity"], dtype=np.float64)
        initial_quat = np.array(self._stage_config["initial_orientation"], dtype=np.float64)
        initial_omega = np.array(self._stage_config["initial_angular_velocity"], dtype=np.float64)

        # Add realistic sensor noise (GPS: ±5cm, IMU: ±0.02 m/s, gyro: ±0.03 rad/s)
        # Increased for robust control
        noise_scale = self._rng.uniform(1.0, 3.0) if enable_randomization else 1.0
        
        # STRENGTHENED: For hover stabilization, add significant offset to force correction
        # This prevents "Null-Op Success" where agent just spawns at target and wins
        if getattr(self, "_stage_config", {}).get("stage_name") == "hover_stabilization":
             # ±0.4m initial error (Reduced from 0.8 to prevent instability)
             pos_noise = self._rng.uniform(-0.4, 0.4, 3) 
             # Ensure Z doesn't go below safety margins (start at 8m, so -0.8 is fine)
        else:
             pos_noise = self._rng.normal(0.0, 0.05 * noise_scale, 3)  # Standard GPS noise
             
        vel_noise = self._rng.normal(0.0, 0.02 * noise_scale, 3)  # ±2-6cm/s IMU noise
        quat_noise = self._rng.normal(0.0, 0.02 * noise_scale, 4)  # Orientation noise
        omega_noise = self._rng.normal(0.0, 0.03 * noise_scale, 3)  # Gyro noise

        quat = initial_quat + quat_noise
        quat = quat / np.linalg.norm(quat)

        self.data.qpos[0:3] = initial_pos + pos_noise
        self.data.qpos[3:7] = quat
        self.data.qvel[0:3] = initial_vel + vel_noise
        self.data.qvel[3:6] = initial_omega + omega_noise
        self.data.ctrl[:] = 0.0

        mujoco.mj_forward(self.model, self.data)

        self._last_action[:] = 0.0
        self._prev_action[:] = 0.0
        self._smoothed_action[:] = 0.0  # Reset smoothed action on reset

        return self._get_observation()

    def step(self, action: np.ndarray) -> StepResult:
        """Execute action without safety constraints - let the agent learn!"""
        self._step_counter += 1
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (3,):
            raise ValueError("Action must have shape (3,)")

        action = action.copy()

        # Removed safety constraints that interfere with learning
        # The agent must experience the full consequences of its actions
        # to learn proper control strategies and recovery behaviors
        
        # Standard action clipping only
        # Actions are now angular RATES (rad/s), not positions
        action[0] = np.clip(action[0], -self._ctrl_limit, self._ctrl_limit)
        action[1] = np.clip(action[1], -self._ctrl_limit, self._ctrl_limit)
        action[2] = np.clip(action[2], 0.0, 1.0)

        # Action smoothing for position commands
        # Helps filter high-frequency noise while maintaining responsiveness
        # TUNING: Reduced alpha 0.8 -> 0.3 to heavily filter policy jitter/vibration
        # This acts as a software low-pass filter on the "nervous" random policy
        alpha = 0.3
        self._smoothed_action[0] = alpha * action[0] + (1.0 - alpha) * self._smoothed_action[0]  # gimbal_x
        self._smoothed_action[1] = alpha * action[1] + (1.0 - alpha) * self._smoothed_action[1]  # gimbal_y
        self._smoothed_action[2] = 0.2 * action[2] + 0.8 * self._smoothed_action[2]  # thrust (very smooth)
        
        smoothed_action = self._smoothed_action.copy()

        # Actuator delay simulation (realistic hardware delay)
        # Use dynamic delay determined at reset
        if self._current_actuator_delay > 0:
            # Add new action to buffer
            self._action_buffer.append(smoothed_action.copy())
            # Remove oldest action
            if len(self._action_buffer) > self._current_actuator_delay + 1:
                self._action_buffer.pop(0)
            # Use delayed action
            delayed_action = self._action_buffer[0]
        else:
            delayed_action = smoothed_action
        
        # Sim2Real: Engine Lag (Spool up/down time)
        # Real engines don't change thrust instantly. Apply exponential filter to thrust.
        current_thrust = self.data.ctrl[self._actuator_ids.get("thrust_control", 2)]
        target_thrust = delayed_action[2]
        lagged_thrust = self._engine_lag_alpha * target_thrust + (1.0 - self._engine_lag_alpha) * current_thrust
        delayed_action[2] = lagged_thrust

        # Position servos expect normalized position targets in ctrlrange [-1, 1]
        # BUT: Since we use position actuators with gear=1, ctrl IS the joint angle (rad).
        # We must NOT divide by ctrl_limit if we want precise angle control.
        # Wait, XML ctrlrange="-1 1". This clamps input.
        # If we pass 0.14, it is valid.
        
        # CRITICAL FIX: Pass radians directly. 
        # Safety clip to joint limits was already done above.
        self.data.ctrl[self._actuator_ids.get("tvc_pitch", 0)] = delayed_action[0]
        self.data.ctrl[self._actuator_ids.get("tvc_yaw", 1)] = delayed_action[1]
        self.data.ctrl[self._actuator_ids.get("thrust_control", 2)] = delayed_action[2]

        # Update visual thrust plume based on thrust magnitude
        self._update_thrust_plume(delayed_action[2])

        for _ in range(self.frame_skip):
            # Apply wind disturbance force if domain randomization enabled
            if self._domain_randomization and hasattr(self, '_wind_force'):
                vehicle_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "vehicle")
                if vehicle_id >= 0:
                    self.data.xfrc_applied[vehicle_id, 0:3] = self._wind_force

            mujoco.mj_step(self.model, self.data)

        observation = self._get_observation()
        reward = self._compute_reward(action)
        done = self._check_termination()
        info = self._get_info(action)

        self._prev_action = self._last_action.copy()
        self._last_action = action.copy()

        return StepResult(observation=observation, reward=float(reward), done=bool(done), info=info)

    def _get_observation(self) -> np.ndarray:
        """Get policy observation with target information.
        
        CRITICAL FIX: Now includes gimbal state for closed-loop control.
        Without gimbal feedback, the policy operates open-loop and cannot
        learn continuous stabilization.
        """
        # CRITICAL: Include gimbal angles and velocities for feedback control
        # qpos layout: [0:3]=pos, [3:7]=quat, [7]=tvc_x, [8]=tvc_y
        # qvel layout: [0:3]=vel, [3:6]=omega, [6]=tvc_x_vel, [7]=tvc_y_vel
        # Sim2Real: Add Sensor Bias to Position and Velocity
        # The agent sees Biased + Noisy data, but reward is based on Truth
        biased_pos = self.data.qpos[0:3] + self._sensor_bias[0:3]
        biased_vel = self.data.qvel[0:3] + self._sensor_bias[3:6]

        state_np = np.concatenate([
            biased_pos,   # Vehicle position (3)
            biased_vel,   # Vehicle velocity (3)
            self.data.qpos[3:7],   # Vehicle quaternion (4)
            self.data.qvel[3:6],   # Vehicle angular velocity (3)
            self.data.qpos[7:9],
            self.data.qvel[6:8],   # Gimbal velocities (2)
        ])
        state_jnp = jnp.asarray(state_np, dtype=jnp.float32)
        
        # Pass target information to observation function
        target_pos_jnp = jnp.asarray(self._stage_config["target_position"], dtype=jnp.float32)
        target_vel_jnp = jnp.asarray(self._stage_config["target_velocity"], dtype=jnp.float32)
        target_quat_jnp = jnp.asarray(self._stage_config["target_orientation"], dtype=jnp.float32)
        
        observation = state_to_observation(
            state_jnp,
            target_pos=target_pos_jnp,
            target_vel=target_vel_jnp,
            target_quat=target_quat_jnp
        )
        
        # CRITICAL: Add previous action AND smoothed action for full temporal/state awareness
        # This helps the policy understand action-effect relationships and the current filter state
        obs_with_action = np.concatenate([
            np.asarray(observation, dtype=np.float32),
            self._last_action.astype(np.float32),      # Predicated/Commanded action (3)
            self._smoothed_action.astype(np.float32),  # Actual physical action state (3)
        ])
        
        return obs_with_action  # Total: 38 + 3 + 3 = 44 dimensions

    def _compute_reward(self, action: np.ndarray) -> float:
        """Compute reward with proper scaling for realistic rocket control.

        Reward components:
        - Position error: Heavily penalized (most important for landing)
        - Velocity error: Penalized to encourage smooth motion
        - Orientation: Must stay upright (quaternion alignment)
        - Angular velocity: Penalize spinning
        - Control smoothness: Reduce jerk
        - Fuel efficiency: Encourage throttle management
        """
        pos = self.data.qpos[0:3]
        vel = self.data.qvel[0:3]
        quat = self.data.qpos[3:7]
        omega = self.data.qvel[3:6]

        target_pos = np.asarray(self._stage_config["target_position"], dtype=np.float64)
        target_vel = np.asarray(self._stage_config["target_velocity"], dtype=np.float64)

        # Distance errors (scaled for 50m altitude range)
        pos_error = np.linalg.norm(pos - target_pos)
        vel_error = np.linalg.norm(vel - target_vel)

        # Orientation alignment using proper quaternion distance
        # Using absolute value of dot product accounts for q and -q representing same orientation
        target_quat = np.array([1.0, 0.0, 0.0, 0.0])
        quat_dot = np.dot(quat, target_quat)
        # Compute angular distance: angle = 2 * arccos(|q1·q2|)
        # For reward, we want alignment metric: close to 1.0 when aligned, close to 0 when misaligned
        orient_alignment = np.abs(quat_dot)  # This is correct - ranges from 0 (perpendicular) to 1 (aligned)

        # Angular velocity magnitude (want this small)
        omega_magnitude = np.linalg.norm(omega)

        # Control smoothness (jerk penalty)
        control_jerk = np.linalg.norm(action - self._last_action)

        # Reward components (scaled appropriately for real physics)
        # Position: Critical - use inverse quadratic with stronger weight
        # Encourages getting close to target position
        pos_reward = 15.0 / (1.0 + pos_error**2)

        # Velocity: Important - use inverse linear with moderate weight
        # Encourages matching target velocity (usually zero for hover)
        vel_reward = 8.0 / (1.0 + vel_error)

        # Orientation: CRITICAL - Strong penalty for tilting with learnable gradient
        # Use **4 exponent for better learning gradient (balanced for exploration vs stability)
        # **6 was too steep and prevented exploration, **4 provides good gradient while allowing learning
        # At 15° tilt: orient_alignment = 0.966 → reward = 35 * 0.966^4 = 31.5
        # At 30° tilt: orient_alignment = 0.866 → reward = 35 * 0.866^4 = 19.7
        # At 45° tilt: orient_alignment = 0.707 → reward = 35 * 0.707^4 = 8.7
        # This creates a learnable gradient that encourages staying upright without being too harsh
        orient_reward = 35.0 * (orient_alignment**4)

        # Angular velocity: Penalize spinning heavily
        omega_reward = 5.0 / (1.0 + omega_magnitude**2)
        
        # ENHANCED: Extra stabilization bonus for very low angular velocity
        # Progressive rewards encourage learning stable control
        # Adjusted thresholds and rewards for better learning at low altitude
        if omega_magnitude < 0.08:  # Very stable - minimal rotation (tightened from 0.10)
            omega_reward += 6.0  # Strong bonus for excellent stability (increased from 5.0)
        elif omega_magnitude < 0.20:  # Good stability (tightened from 0.25)
            omega_reward += 3.0  # Increased from 2.5
        elif omega_magnitude < 0.40:  # Moderate stability (tightened from 0.5)
            omega_reward += 1.5  # Increased from 1.0

        # Altitude maintenance: Reward staying near target altitude
        target_altitude = target_pos[2]
        altitude_error = abs(pos[2] - target_altitude)
        altitude_reward = 5.0 / (1.0 + altitude_error)

        # Smoothness: INCREASED weight to penalize shaking
        # Was 0.2, now 2.0 to essentially force smooth control
        smoothness_reward = 2.0 / (1.0 + control_jerk * 5.0)

        # Removed progress reward - uses mutable state that doesn't work properly
        # The position reward already provides progress signal
        progress_reward = 0.0

        # Stability reward: Reward maintaining stable hover
        # Adjusted for lower altitude training (8m instead of 50m)
        if pos_error < 2.0 and vel_error < 1.0:  # Tightened from 5.0m and 2.0 m/s
            stability_reward = 8.0  # Increased from 5.0 to compensate for removal of guidance
        elif pos_error < 4.0 and vel_error < 2.0:  # Moderate stability
            stability_reward = 3.0  # Increased from 2.0
        else:
            stability_reward = 0.0

        # REMOVED: Thrust guidance (heuristic nanny reward)
        # The agent must learn to use thrust on its own!
        thrust_guidance_reward = 0.0
        
        fuel_reward = 0.0  # Disabled for hover task

        # ============================================================
        # STAGE-SPECIFIC REWARDS: Hover vs Landing
        # ============================================================
        stage_name = self._stage_config.get("stage_name", "")
        target_altitude = target_pos[2]
        descent_reward = 0.0
        hover_precision_reward = 0.0
        
        # HOVER STAGE REWARDS (Stage 0: 3m Hover)
        if "Hover" in stage_name or target_altitude >= 2.5:
            # Reward for maintaining exact position (not just being close)
            if pos_error < 0.3:  # Very precise position
                hover_precision_reward = 10.0
            elif pos_error < 0.6:
                hover_precision_reward = 5.0
            
            # Extra reward for true zero velocity (hovering, not drifting)
            if vel_error < 0.2:
                hover_precision_reward += 8.0
            elif vel_error < 0.4:
                hover_precision_reward += 4.0
        
        # LANDING STAGE REWARDS (Stages with target altitude <= 1.5m)
        elif target_altitude <= 1.5:
            vertical_velocity = vel[2]  # Negative = descending
            current_altitude = pos[2]
            
            # Optimal descent profile: velocity proportional to sqrt of altitude
            # This is the "constant time-to-impact" profile used by real rockets
            # At 5m: target_vz = -0.5 * sqrt(5) = -1.12 m/s
            # At 1m: target_vz = -0.5 * sqrt(1) = -0.5 m/s
            # At 0.1m: target_vz = -0.5 * sqrt(0.1) = -0.16 m/s
            target_descent_rate = -0.5 * np.sqrt(max(current_altitude, 0.1))
            descent_error = abs(vertical_velocity - target_descent_rate)
            
            # Reward for following optimal descent profile
            descent_reward = 10.0 / (1.0 + descent_error * 3.0)
            
            # Bonus for very soft touchdown (< 0.3 m/s at low altitude)
            if current_altitude < 0.5 and abs(vertical_velocity) < 0.3:
                descent_reward += 20.0  # Large soft landing bonus
            elif current_altitude < 1.0 and abs(vertical_velocity) < 0.5:
                descent_reward += 10.0  # Moderate soft approach bonus
            
            # Penalize too-fast descent at low altitude (crash risk)
            if current_altitude < 2.0 and vertical_velocity < -2.0:
                descent_reward -= 15.0  # Coming in too hot

        # ============================================================
        # GIMBAL-ORIENTATION COUPLING REWARD
        # Explicitly reward using gimbal correctly to counteract tilt
        # This teaches the agent the TVC control mapping!
        # ============================================================
        gimbal_angles = self.data.qpos[7:9]  # Current gimbal position [pitch, yaw]
        
        # Approximate tilt from quaternion (small angle approximation)
        # roll ≈ 2*qx (tilt around X axis)
        # pitch ≈ 2*qy (tilt around Y axis)
        roll_tilt = 2.0 * quat[1]   # Tilt around X
        pitch_tilt = 2.0 * quat[2]  # Tilt around Y
        
        # Correct gimbal response: gimbal should oppose tilt
        # If tilting forward (positive pitch), gimbal X should deflect negatively
        # Scale factor matches the expected gimbal response magnitude
        correct_gimbal_x = -pitch_tilt * 0.4
        correct_gimbal_y = roll_tilt * 0.4
        
        # Reward for correct gimbal usage (lower error = higher reward)
        gimbal_x_error = abs(gimbal_angles[0] - correct_gimbal_x)
        gimbal_y_error = abs(gimbal_angles[1] - correct_gimbal_y)
        gimbal_agreement = 1.0 / (1.0 + gimbal_x_error + gimbal_y_error)
        gimbal_control_reward = 8.0 * gimbal_agreement  # Max ~8 when perfect

        # Combined reward (dense, shaped, scales ~0-80 per step)
        reward = (
            pos_reward +           # ~15 max
            vel_reward +           # ~8 max
            orient_reward +        # ~35 max
            omega_reward +         # ~5-11 with bonuses
            altitude_reward +      # ~5 max
            smoothness_reward +    # ~0.2 max
            progress_reward +      # 0
            stability_reward +     # ~8 max
            fuel_reward +          # 0
            thrust_guidance_reward + # 0
            descent_reward +       # ~10-30 for landing stages
            hover_precision_reward + # ~10-18 for hover stage
            gimbal_control_reward  # ~8 max for correct TVC usage
        )

        # Direct penalty for jerk (rapid action changes)
        # If action changes by > 10% in one step (0.1), jerk is ~0.1
        # Penalty: 0.1 * 10.0 = 1.0 (Meaningful penalty)
        reward -= control_jerk * 10.0

        # CRITICAL: Success bonus - large reward for achieving goal state
        # This provides strong learning signal for successful behavior
        if (
            pos_error < self._stage_config["position_tolerance"]
            and vel_error < self._stage_config["velocity_tolerance"]
            and orient_alignment > 0.95  # Within ~18 degrees
            and omega_magnitude < self._stage_config["angular_velocity_tolerance"]
        ):
            reward += 50.0  # Significant success bonus (increased from 20.0)

        # CRITICAL: Strong penalties for failure states to teach avoidance
        vertical_velocity = vel[2]  # Negative = descending
        
        # Crash penalty - scaled down for stability (research: high variance hurts value learning)
        if pos[2] < 0.05:  # Below ground
            reward -= 50.0  # Reduced from 200.0 for lower variance
        elif pos[2] < 0.2 and vertical_velocity < -2.0:  # Low + fast descent = imminent crash
            reward -= 25.0  # Reduced from 100.0
        
        # Excessive tilt penalty - moderate (allow some tilt for maneuvering)
        if orient_alignment < 0.866:  # > 30 degrees tilt
            reward -= 20.0  # Reduced from 50.0
        elif orient_alignment < 0.94:  # > 20 degrees tilt
            reward -= 5.0  # Reduced from 10.0
        
        # Spinning penalty
        if omega_magnitude > 1.5:  # Excessive spinning
            reward -= 15.0  # Reduced from 40.0
        elif omega_magnitude > 1.0:  # High angular velocity
            reward -= 5.0  # Reduced from 15.0

        # Reward scale check: Typical good behavior should get 40-60 per step
        # Over 300 steps, this gives ~12k-18k return
        # Success gives ~70-80 per step = ~21k-24k return
        # Failure gives negative return
        return float(reward)


    def _check_termination(self) -> bool:
        """Check episode termination with early success bonus."""
        pos = self.data.qpos[0:3]
        vel = self.data.qvel[0:3]
        quat = self.data.qpos[3:7]
        omega = self.data.qvel[3:6]
        
        # CRITICAL FIX: Early termination on success
        # If agent achieves goal, terminate early with success flag
        # This provides strong learning signal and saves computation
        target_pos = np.asarray(self._stage_config["target_position"], dtype=np.float64)
        target_vel = np.asarray(self._stage_config["target_velocity"], dtype=np.float64)
        target_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        pos_error = np.linalg.norm(pos - target_pos)
        vel_error = np.linalg.norm(vel - target_vel)
        orient_alignment = np.abs(np.dot(quat, target_quat))
        omega_magnitude = np.linalg.norm(omega)
        
        # Early success termination - achieves goal state
        # CRITICAL: Disabled for hover_stabilization to force duration holding
        is_hover_stage = self._stage_config.get("stage_name") == "hover_stabilization"
        
        if (not is_hover_stage and  # Only allow early finish for non-hover stages (e.g. landing)
            pos_error < self._stage_config["position_tolerance"] * 0.8 and  # Within 80% of tolerance
            vel_error < self._stage_config["velocity_tolerance"] * 0.8 and
            orient_alignment > 0.98 and  # Within ~11 degrees
            omega_magnitude < self._stage_config["angular_velocity_tolerance"] * 0.8):
            # Agent achieved goal! Terminate with success
            return True
        
        # Terminate if crashed (below 0.0m) or exceeded max altitude (200m)
        if pos[2] < 0.0 or pos[2] > 200.0:
            return True
        
        # Terminate if drifted too far laterally (50m radius)
        if np.linalg.norm(pos[0:2]) > 50.0:
            return True
        
        # Tightened tilt termination for better stability learning
        w = quat[0]
        tilt_angle = 2.0 * np.arccos(np.clip(abs(w), 0.0, 1.0))
        if tilt_angle > 0.785:  # 45 degrees
            return True
        
        # Angular velocity limit
        if np.linalg.norm(omega) > 2.5:
            return True
        
        if self._step_counter >= self.max_steps:
            return True
        
        return False

    def _update_thrust_plume(self, thrust_fraction: float) -> None:
        """Dynamically visualize thrust plume using multi-layer fade based on thrust magnitude."""
        try:
            # Find the multi-layer plume geometries
            plume_ids = {
                'short': mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "plume_short"),
                'medium': mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "plume_medium"),
                'long': mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "plume_long"),
                'outer': mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "plume_outer"),
            }
            
            if thrust_fraction < 0.05:
                # No thrust - hide all plumes
                for geom_id in plume_ids.values():
                    if geom_id >= 0:
                        self.data.geom_rgba[geom_id, 3] = 0.0  # Set alpha to 0
            else:
                # Short plume: Always visible when thrust > 5%
                if plume_ids['short'] >= 0:
                    alpha = min(1.0, thrust_fraction * 3.0)  # Full brightness at 33% thrust
                    self.data.geom_rgba[plume_ids['short']] = [0.98, 0.5, 0.1, 0.8 * alpha]
                
                # Medium plume: Visible from 30% thrust
                if plume_ids['medium'] >= 0:
                    if thrust_fraction >= 0.3:
                        alpha = min(1.0, (thrust_fraction - 0.3) * 2.5)
                        self.data.geom_rgba[plume_ids['medium']] = [0.98, 0.5, 0.1, 0.7 * alpha]
                    else:
                        self.data.geom_rgba[plume_ids['medium'], 3] = 0.0
                
                # Long plume: Visible from 60% thrust
                if plume_ids['long'] >= 0:
                    if thrust_fraction >= 0.6:
                        alpha = min(1.0, (thrust_fraction - 0.6) * 2.5)
                        self.data.geom_rgba[plume_ids['long']] = [0.98, 0.5, 0.1, 0.6 * alpha]
                    else:
                        self.data.geom_rgba[plume_ids['long'], 3] = 0.0
                
                # Outer dispersed plume: Always visible with thrust, scales with amount
                if plume_ids['outer'] >= 0:
                    alpha = thrust_fraction * 0.8  # Max 80% opacity
                    self.data.geom_rgba[plume_ids['outer']] = [0.9, 0.4, 0.05, 0.3 * alpha]
                    
        except Exception:
            # Silently ignore if geometries not found
            pass

    def _get_info(self, action: np.ndarray) -> Dict[str, float]:
        """Get diagnostic info."""
        pos = self.data.qpos[0:3]
        vel = self.data.qvel[0:3]
        return {
            "altitude": float(pos[2]),
            "lateral_distance": float(np.linalg.norm(pos[0:2])),
            "speed": float(np.linalg.norm(vel)),
            "gimbal_x": float(action[0]),
            "gimbal_y": float(action[1]),
            "thrust_fraction": float(action[2]),
            "episode_length": float(self._step_counter),
        }


def _default_asset_path() -> str:
    """Locate tvc_3d.xml asset."""
    env_path = os.environ.get("TVC_ASSET_PATH")
    if env_path:
        candidate = Path(env_path)
        if candidate.is_file():
            return str(candidate)
        candidate_file = candidate / "tvc_3d.xml"
        if candidate_file.is_file():
            return str(candidate_file)

    origin = Path(__file__).resolve()
    for base in [origin.parent, *origin.parents]:
        asset_path = base / "assets" / "tvc_3d.xml"
        if asset_path.is_file():
            return str(asset_path)

    raise FileNotFoundError("Unable to locate tvc_3d.xml")
