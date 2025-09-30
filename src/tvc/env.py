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
        ctrl_limit: float = 0.3,
        max_steps: int = 2000,
        seed: int | None = None,
    ) -> None:
        asset_path = model_path or _default_asset_path()
        self.model = mujoco.MjModel.from_xml_path(asset_path)
        self.data = mujoco.MjData(self.model)

        self._ctrl_limit = float(ctrl_limit)
        self.frame_skip = max(int(dt / self.model.opt.timestep), 1)
        self.max_steps = int(max_steps)
        self._rng = np.random.default_rng(seed)
        self._step_counter = 0

        # Reward normalization
        self._reward_count = 0.0
        self._reward_mean = 0.0
        self._reward_m2 = 0.0
        self._reward_norm_warmup = 64
        self._reward_norm_clip = 6.0

        # Action tracking
        self._last_action = np.zeros(3, dtype=np.float64)
        self._prev_action = np.zeros(3, dtype=np.float64)

        # Stage configuration
        self._stage_config = {
            "target_position": [0.0, 0.0, 8.0],
            "target_orientation": [1.0, 0.0, 0.0, 0.0],
            "target_velocity": [0.0, 0.0, 0.0],
            "target_angular_velocity": [0.0, 0.0, 0.0],
            "initial_position": [0.0, 0.0, 8.0],
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
        """Reset environment."""
        self._step_counter = 0
        mujoco.mj_resetData(self.model, self.data)

        initial_pos = np.array(self._stage_config["initial_position"], dtype=np.float64)
        initial_vel = np.array(self._stage_config["initial_velocity"], dtype=np.float64)
        initial_quat = np.array(self._stage_config["initial_orientation"], dtype=np.float64)
        initial_omega = np.array(self._stage_config["initial_angular_velocity"], dtype=np.float64)

        # Add noise
        pos_noise = self._rng.normal(0.0, 0.4, 3)
        vel_noise = self._rng.normal(0.0, 0.2, 3)
        quat_noise = self._rng.normal(0.0, 0.05, 4)
        omega_noise = self._rng.normal(0.0, 0.1, 3)

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
        """Execute action."""
        self._step_counter += 1
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (3,):
            raise ValueError("Action must have shape (3,)")

        action = action.copy()
        action[0] = np.clip(action[0], -self._ctrl_limit, self._ctrl_limit)
        action[1] = np.clip(action[1], -self._ctrl_limit, self._ctrl_limit)
        action[2] = np.clip(action[2], 0.0, 1.0)

        self.data.ctrl[self._actuator_ids.get("tvc_pitch", 0)] = action[0] / max(self._ctrl_limit, 1e-6)
        self.data.ctrl[self._actuator_ids.get("tvc_yaw", 1)] = action[1] / max(self._ctrl_limit, 1e-6)
        self.data.ctrl[self._actuator_ids.get("thrust_control", 2)] = action[2]

        for _ in range(self.frame_skip):
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
        """Compute reward."""
        pos = self.data.qpos[0:3]
        vel = self.data.qvel[0:3]
        quat = self.data.qpos[3:7]
        omega = self.data.qvel[3:6]

        target_pos = np.asarray(self._stage_config["target_position"], dtype=np.float64)
        target_vel = np.asarray(self._stage_config["target_velocity"], dtype=np.float64)

        pos_error = np.linalg.norm(pos - target_pos)
        vel_error = np.linalg.norm(vel - target_vel)
        target_quat = np.array([1.0, 0.0, 0.0, 0.0])
        orient_alignment = np.abs(np.dot(quat, target_quat))
        omega_penalty = np.exp(-0.5 * np.linalg.norm(omega))
        smoothness_reward = np.exp(-2.0 * np.linalg.norm(action - self._last_action))
        fuel_efficiency = 1.0 - 0.3 * action[2]

        reward = (
            0.4 * np.exp(-2.0 * pos_error)
            + 0.2 * np.exp(-1.0 * vel_error)
            + 0.2 * orient_alignment**2
            + 0.1 * omega_penalty
            + 0.05 * smoothness_reward
            + 0.05 * fuel_efficiency
        )

        if (
            pos_error < self._stage_config["position_tolerance"]
            and vel_error < self._stage_config["velocity_tolerance"]
        ):
            reward += self._stage_config["tolerance_bonus"]

        normalised, _ = self._normalise_reward(reward)
        return normalised

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
        """Check episode termination."""
        pos = self.data.qpos[0:3]
        if pos[2] < 0.1 or pos[2] > 25.0:
            return True
        if np.linalg.norm(pos[0:2]) > 15.0:
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