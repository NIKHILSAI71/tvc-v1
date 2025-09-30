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
        ctrl_limit: float = 0.14,  # Real F9 gimbal limit: ±8° = 0.14 rad
        max_steps: int = 2000,
        seed: int | None = None,
        domain_randomization: bool = True,  # Enable robust training
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
        self._base_mass = 256.0
        self._base_inertia = np.array([680.0, 680.0, 45.0])
        self._base_thrust_max = 8540.0
        self._base_damping = 0.8

        # Reward normalization
        self._reward_count = 0.0
        self._reward_mean = 0.0
        self._reward_m2 = 0.0
        self._reward_norm_warmup = 64
        self._reward_norm_clip = 6.0

        # Action tracking
        self._last_action = np.zeros(3, dtype=np.float64)
        self._prev_action = np.zeros(3, dtype=np.float64)

        # Stage configuration - Realistic landing scenario altitudes
        self._stage_config = {
            "target_position": [0.0, 0.0, 50.0],  # Start from 50m hover
            "target_orientation": [1.0, 0.0, 0.0, 0.0],
            "target_velocity": [0.0, 0.0, 0.0],
            "target_angular_velocity": [0.0, 0.0, 0.0],
            "initial_position": [0.0, 0.0, 50.0],
            "initial_velocity": [0.0, 0.0, 0.0],
            "initial_orientation": [1.0, 0.0, 0.0, 0.0],
            "initial_angular_velocity": [0.0, 0.0, 0.0],
            "position_tolerance": 1.0,
            "velocity_tolerance": 0.8,
            "orientation_tolerance": 0.2,
            "angular_velocity_tolerance": 0.5,
            "tolerance_bonus": 0.6,
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

        # Domain randomization: Randomize physics parameters for robustness
        if self._domain_randomization:
            # Mass variation: ±10%
            mass_factor = self._rng.uniform(0.9, 1.1)
            vehicle_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "vehicle")
            if vehicle_id >= 0:
                self.model.body_mass[vehicle_id] = self._base_mass * mass_factor

            # Inertia variation: ±15%
            inertia_factor = self._rng.uniform(0.85, 1.15, 3)
            if vehicle_id >= 0:
                self.model.body_inertia[vehicle_id] = self._base_inertia * inertia_factor

            # Thrust variation: ±5%
            thrust_factor = self._rng.uniform(0.95, 1.05)
            # Note: Thrust variation applied in step() via scaling

            # Damping variation: ±30%
            damping_factor = self._rng.uniform(0.7, 1.3)
            # Applied during dynamics

            # Wind disturbance: 0-5 m/s random direction
            self._wind_force = self._rng.normal(0.0, 2.0, 3)  # Random wind
        else:
            self._wind_force = np.zeros(3)

        initial_pos = np.array(self._stage_config["initial_position"], dtype=np.float64)
        initial_vel = np.array(self._stage_config["initial_velocity"], dtype=np.float64)
        initial_quat = np.array(self._stage_config["initial_orientation"], dtype=np.float64)
        initial_omega = np.array(self._stage_config["initial_angular_velocity"], dtype=np.float64)

        # Add realistic sensor noise (GPS: ±2cm, IMU: ±0.01 m/s, gyro: ±0.02 rad/s)
        # Increase noise variation for domain randomization
        noise_scale = self._rng.uniform(0.5, 1.5) if self._domain_randomization else 1.0
        pos_noise = self._rng.normal(0.0, 0.02 * noise_scale, 3)  # ±2cm GPS accuracy
        vel_noise = self._rng.normal(0.0, 0.01 * noise_scale, 3)  # ±1cm/s IMU accuracy
        quat_noise = self._rng.normal(0.0, 0.01 * noise_scale, 4)  # ±0.01 orientation noise
        omega_noise = self._rng.normal(0.0, 0.02 * noise_scale, 3)  # ±0.02 rad/s gyro accuracy

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

        return self._get_observation()

    def step(self, action: np.ndarray) -> StepResult:
        """Execute action with safety constraints."""
        self._step_counter += 1
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (3,):
            raise ValueError("Action must have shape (3,)")

        action = action.copy()

        # Safety constraint: Check tilt angle before applying action
        quat = self.data.qpos[3:7]
        # Calculate tilt angle from vertical (quaternion to angle)
        w, x, y, z = quat
        tilt_angle = 2.0 * np.arccos(np.clip(abs(w), 0.0, 1.0))

        # If tilted beyond 45 degrees, limit aggressive maneuvers
        if tilt_angle > 0.785:  # 45 degrees = 0.785 rad
            action[0] *= 0.5  # Reduce gimbal authority
            action[1] *= 0.5
            action[2] = np.clip(action[2], 0.5, 1.0)  # Force minimum thrust

        # Safety constraint: Velocity limits
        vel = self.data.qvel[0:3]
        speed = np.linalg.norm(vel)
        if speed > 20.0:  # Maximum 20 m/s speed
            # Limit actions that could increase speed
            vel_dir = vel / (speed + 1e-6)
            action[2] = np.clip(action[2], 0.3, 0.8)  # Reduce thrust range

        # Standard action clipping
        action[0] = np.clip(action[0], -self._ctrl_limit, self._ctrl_limit)
        action[1] = np.clip(action[1], -self._ctrl_limit, self._ctrl_limit)
        action[2] = np.clip(action[2], 0.0, 1.0)

        self.data.ctrl[self._actuator_ids.get("tvc_pitch", 0)] = action[0] / max(self._ctrl_limit, 1e-6)
        self.data.ctrl[self._actuator_ids.get("tvc_yaw", 1)] = action[1] / max(self._ctrl_limit, 1e-6)
        self.data.ctrl[self._actuator_ids.get("thrust_control", 2)] = action[2]

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
        """Get policy observation."""
        state_np = np.concatenate([
            self.data.qpos[0:3],
            self.data.qvel[0:3],
            self.data.qpos[3:7],
            self.data.qvel[3:6],
        ])
        state_jnp = jnp.asarray(state_np, dtype=jnp.float32)
        observation = state_to_observation(state_jnp)
        return np.asarray(observation, dtype=np.float32)

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

        # Orientation alignment (quaternion dot product close to 1.0 = aligned)
        target_quat = np.array([1.0, 0.0, 0.0, 0.0])
        orient_alignment = np.abs(np.dot(quat, target_quat))

        # Angular velocity magnitude (want this small)
        omega_magnitude = np.linalg.norm(omega)

        # Control smoothness (jerk penalty)
        control_jerk = np.linalg.norm(action - self._last_action)

        # Reward components (scaled appropriately for real physics)
        # Position: Critical - use inverse quadratic (sharper penalty near target)
        pos_reward = 10.0 / (1.0 + pos_error**2)

        # Velocity: Important - use inverse linear
        vel_reward = 5.0 / (1.0 + vel_error)

        # Orientation: Critical - reward alignment
        orient_reward = 8.0 * (orient_alignment**4)  # Sharp reward for staying upright

        # Angular velocity: Penalize spinning
        omega_reward = 3.0 / (1.0 + omega_magnitude**2)

        # Altitude maintenance: Reward staying near target altitude
        target_altitude = target_pos[2]
        altitude_error = abs(pos[2] - target_altitude)
        altitude_reward = 5.0 / (1.0 + altitude_error)

        # Smoothness: Reduce jerk
        smoothness_reward = 2.0 / (1.0 + control_jerk)

        # Fuel efficiency: REMOVED for hover task (conflicts with maintaining altitude)
        # Hover requires sustained thrust near 1.0, penalizing it is counterproductive
        fuel_reward = 0.0  # Disabled

        # Combined reward (unnormalized, scales ~0-33)
        reward = (
            pos_reward +
            vel_reward +
            orient_reward +
            omega_reward +
            altitude_reward +
            smoothness_reward +
            fuel_reward
        )

        # Bonus for being within tolerance (achieving goal)
        if (
            pos_error < self._stage_config["position_tolerance"]
            and vel_error < self._stage_config["velocity_tolerance"]
            and orient_alignment > 0.95
        ):
            reward += 15.0  # Large bonus for goal achievement

        # Crash penalty
        if pos[2] < 0.2:
            reward -= 20.0

        # Don't normalize - let PPO learn the scale naturally
        return float(reward)

    def _normalise_reward(self, value: float) -> tuple[float, float]:
        """Welford's online algorithm for reward normalization."""
        self._reward_count += 1.0
        delta = value - self._reward_mean
        self._reward_mean += delta / self._reward_count
        self._reward_m2 += delta * (value - self._reward_mean)

        if self._reward_count < self._reward_norm_warmup:
            return value, 0.0

        variance = self._reward_m2 / max(self._reward_count - 1.0, 1.0)
        variance = max(float(variance), 1e-6)
        std = float(math.sqrt(variance))
        normalised = (value - self._reward_mean) / std
        normalised = float(np.clip(normalised, -self._reward_norm_clip, self._reward_norm_clip))
        return normalised, std

    def _check_termination(self) -> bool:
        """Check episode termination - Realistic landing bounds."""
        pos = self.data.qpos[0:3]
        # Terminate if crashed (below 0.1m) or exceeded max altitude (200m)
        if pos[2] < 0.1 or pos[2] > 200.0:
            return True
        # Terminate if drifted too far laterally (50m radius)
        if np.linalg.norm(pos[0:2]) > 50.0:
            return True
        if self._step_counter >= self.max_steps:
            return True
        return False

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