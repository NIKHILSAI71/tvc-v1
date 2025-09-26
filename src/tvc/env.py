"""Environment utilities for the 2D thrust-vector-control research platform.

This module exposes a lightweight MuJoCo wrapper that keeps the vehicle within a
2D plane while enabling accurate rigid-body dynamics. It also provides helpers
for interacting with the MJX GPU-accelerated backend so that large training
batches can be simulated efficiently when curriculum schedules demand many
parallel scenarios.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import math

import mujoco as _mujoco
import numpy as np
from mujoco import mjx

from .dynamics import RocketParams

mujoco: Any = _mujoco


@dataclass
class StepResult:
    """Container for simulation feedback produced by :class:`Tvc2DEnv`.

    Args:
        observation: Vector fed into control policies (thrust, angle, angular rate).
        reward: Scalar reward promoting stable, smooth flight.
        done: Boolean terminating flag for curriculum rollouts.
        info: Extra diagnostics (e.g., jerk metrics, constraint residuals).
    """

    observation: np.ndarray
    reward: float
    done: bool
    info: Dict[str, float]


class Tvc2DEnv:
    """MuJoCo-backed planar rocket environment with thrust vector control.

    Args:
        model_path: Path to the MJCF model. Defaults to the packaged ``assets/tvc_2d.xml``.
        dt: Integration step used when aggregating multiple MuJoCo sub-steps per RL step.
        ctrl_limit: Maximum absolute displacement (meters) for the TVC carriage on each axis.
        max_steps: Safety guard for truncating long episodes when curriculum resets are needed.
        seed: Seed for the local RNG injected into disturbance sampling.
    """

    def __init__(
        self,
        model_path: str | None = None,
        dt: float = 0.02,
        ctrl_limit: float = 0.28,
        max_steps: int = 2000,
        seed: int | None = None,
    ) -> None:
        asset_path = model_path or _default_asset_path()
        self.model = mujoco.MjModel.from_xml_path(asset_path)
        self.data = mujoco.MjData(self.model)
        self.ctrl_limit = float(ctrl_limit)
        self.frame_skip = max(int(dt / self.model.opt.timestep), 1)
        self.max_steps = int(max_steps)
        self._rng = np.random.default_rng(seed)
        self._step_counter = 0
        # Reward statistics for on-the-fly normalisation.
        self._reward_count = 0.0
        self._reward_mean = 0.0
        self._reward_m2 = 0.0
        self._reward_norm_warmup = 64
        self._reward_norm_clip = 6.0
        mujoco.mj_forward(self.model, self.data)

    def apply_rocket_params(self, params: RocketParams | None) -> None:
        """Reconfigures the MuJoCo model to match supplied physical parameters."""

        if params is None:
            return

        mjt_obj = getattr(mujoco, "mjtObj")
        vehicle_id = int(mujoco.mj_name2id(self.model, mjt_obj.mjOBJ_BODY, "vehicle"))
        if vehicle_id >= 0:
            self.model.body_mass[vehicle_id] = params.mass
            inertia = np.array([
                max(params.inertia, 1e-3),
                max(params.inertia, 1e-3),
                max(params.inertia * 0.5, 1e-3),
            ])
            self.model.body_inertia[vehicle_id] = inertia

        gravity = np.array([0.0, 0.0, -params.gravity], dtype=np.float64)
        self.model.opt.gravity[:] = gravity

        joint_limit = min(self.ctrl_limit, max(params.arm * 0.25, 0.05))
        for joint_name in ("tvc_x", "tvc_y"):
            joint_id = int(mujoco.mj_name2id(self.model, mjt_obj.mjOBJ_JOINT, joint_name))
            if joint_id >= 0:
                self.model.jnt_range[joint_id] = np.array([-joint_limit, joint_limit])
        self.ctrl_limit = float(joint_limit)
        mujoco.mj_forward(self.model, self.data)

    def reset(self, disturbance_scale: float = 1.0) -> np.ndarray:
        """Reinitialises the simulation state with configurable disturbances.

        Args:
            disturbance_scale: Amplitude applied to random attitude and velocity perturbations.

        Returns:
            Observation vector used as the initial input for the controller.
        """

        mujoco.mj_resetData(self.model, self.data)
        # Initial altitude and alignment.
        self.data.qpos[2] = 3.5
        self.data.qvel[:] = 0.0

        # Inject bounded disturbances to mimic pre-launch uncertainties.
        angle_disturb = self._rng.normal(0.0, 0.02 * disturbance_scale)
        mujoco.mju_axisAngle2Quat(self.data.qpos[3:7], np.array([1.0, 0.0, 0.0]), angle_disturb)
        self.data.qvel[3:6] = self._rng.normal(0.0, 0.5 * disturbance_scale, size=3)

        # Random lateral offsets.
        self.data.qpos[0] = self._rng.normal(0.0, 0.3 * disturbance_scale)
        self.data.qvel[0] = self._rng.normal(0.0, 0.5 * disturbance_scale)

        mujoco.mj_forward(self.model, self.data)
        self._step_counter = 0
        return self._get_observation()

    def step(self, action: np.ndarray, thrust_command: float | None = None) -> StepResult:
        """Steps the environment using planar TVC offsets.

        Args:
            action: Array ``[x, y]`` specifying the lateral displacement of the nozzle (meters).
            thrust_command: Optional scalar throttle command (0–1). When omitted a hover-hold
                controller modulates thrust to stabilise altitude and pitch.

        Returns:
            A :class:`StepResult` capturing policy inputs and physics-derived metrics.
        """

        assert action.shape == (2,), "Action must provide x and y offsets."
        ctrl = self.data.ctrl
        ctrl[1:3] = np.clip(action, -self.ctrl_limit, self.ctrl_limit)

        if thrust_command is None:
            ctrl[0] = self._hover_thrust()
        else:
            ctrl[0] = float(np.clip(thrust_command, -1.0, 1.0))

        # Integrate using smaller MuJoCo steps for stability.
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        observation = self._get_observation()
        reward, info = self._reward()
        self._step_counter += 1

        done = bool(
            self._step_counter >= self.max_steps
            or self.data.qpos[2] < 0.5
            or abs(self._pitch()) > 0.6
        )
        return StepResult(observation=observation, reward=reward, done=done, info=info)

    def mjx_state(self) -> Tuple[mjx.Model, mjx.Data]:
        """Exports an MJX-ready model/data pair anchored to the current MuJoCo state."""

        mjx_model = mjx.put_model(self.model)
        mjx_data = mjx.make_data(mjx_model)
        mjx_data = mjx_data.replace(
            qpos=np.array(self.data.qpos),
            qvel=np.array(self.data.qvel),
            ctrl=np.array(self.data.ctrl),
        )
        return mjx_model, mjx_data

    def _hover_thrust(self) -> float:
        height_error = 4.0 - float(self.data.qpos[2])
        vertical_vel = float(self.data.qvel[2])
        pitch_error = self._pitch()
        pitch_rate = self._pitch_rate()
        # Simple PD to keep the rocket upright and hovering.
        thrust = 0.6 + 0.08 * height_error - 0.015 * vertical_vel - 0.4 * pitch_error - 0.03 * pitch_rate
        return float(np.clip(thrust, -1.0, 1.0))

    def _get_observation(self) -> np.ndarray:
        thrust = float(self.data.ctrl[0])
        pitch = self._pitch()
        pitch_rate = self._pitch_rate()
        lateral = float(self.data.qpos[0])
        lateral_vel = float(self.data.qvel[0])
        altitude = float(self.data.qpos[2])
        vertical_vel = float(self.data.qvel[2])
        return np.array(
            [thrust, pitch, pitch_rate, lateral, lateral_vel, altitude, vertical_vel],
            dtype=np.float32,
        )

    def _pitch(self) -> float:
        quat = np.array(self.data.qpos[3:7])
        qw, qx, qy, qz = quat
        sin_pitch = 2.0 * (qw * qy - qz * qx)
        sin_pitch = np.clip(sin_pitch, -1.0, 1.0)
        return float(np.arcsin(sin_pitch))

    def _pitch_rate(self) -> float:
        # Angular velocity around Y-axis.
        return float(self.data.qvel[4])

    def _reward(self) -> Tuple[float, Dict[str, float]]:
        pitch = self._pitch()
        pitch_rate = self._pitch_rate()
        lateral = float(self.data.qpos[0])
        lateral_vel = float(self.data.qvel[0])
        altitude_error = 4.0 - float(self.data.qpos[2])
        vertical_vel = float(self.data.qvel[2])
        jerk = float(np.linalg.norm(self.data.qacc[:3]))
        ctrl_jitter = float(np.linalg.norm(np.diff(self.data.ctrl)))
        ctrl_magnitude = float(np.linalg.norm(self.data.ctrl[1:3]))

        # Normalised quadratic costs using scenario-appropriate tolerances.
        def _quad(value: float, tolerance: float) -> float:
            scale = max(tolerance, 1e-6)
            return (value / scale) ** 2

        pitch_cost = _quad(pitch, 0.18)
        pitch_rate_cost = _quad(pitch_rate, 0.6)
        lateral_cost = _quad(lateral, 1.0)
        lateral_vel_cost = _quad(lateral_vel, 1.2)
        altitude_cost = _quad(altitude_error, 0.8)
        vertical_vel_cost = _quad(vertical_vel, 1.4)
        jerk_cost = _quad(jerk, 40.0)
        ctrl_jitter_cost = _quad(ctrl_jitter, 0.25)
        ctrl_mag_cost = _quad(ctrl_magnitude, 0.28)

        penalty = (
            0.38 * pitch_cost
            + 0.16 * pitch_rate_cost
            + 0.26 * lateral_cost
            + 0.12 * lateral_vel_cost
            + 0.34 * altitude_cost
            + 0.22 * vertical_vel_cost
            + 0.05 * jerk_cost
            + 0.04 * ctrl_jitter_cost
            + 0.03 * ctrl_mag_cost
        )

        # Smooth positive shaping encouraging hover stability and gentle control.
        stability_bonus = float(
            0.45
            * math.exp(-0.5 * ((pitch / 0.11) ** 2 + (lateral / 0.55) ** 2))
        )
        hover_bonus = float(
            0.35 * math.exp(-0.5 * ((altitude_error / 0.35) ** 2 + (vertical_vel / 0.45) ** 2))
        )
        attitude_bonus = float(0.25 * math.exp(-0.5 * (pitch_rate / 0.35) ** 2))
        action_bonus = float(0.12 * math.exp(-0.5 * (ctrl_magnitude / 0.18) ** 2))

        raw_reward = 1.0 - penalty + stability_bonus + hover_bonus + attitude_bonus + action_bonus
        raw_reward = float(np.clip(raw_reward, -5.0, 3.0))
        normalised_reward, reward_std = self._normalise_reward(raw_reward)

        return normalised_reward, {
            "pitch_error": pitch,
            "pitch_rate": pitch_rate,
            "lateral": lateral,
            "lateral_vel": lateral_vel,
            "altitude_error": altitude_error,
            "vertical_velocity": vertical_vel,
            "jerk": jerk,
            "ctrl_jitter": ctrl_jitter,
            "ctrl_magnitude": ctrl_magnitude,
            "reward_penalty": penalty,
            "stability_bonus": stability_bonus,
            "hover_bonus": hover_bonus,
            "attitude_bonus": attitude_bonus,
            "action_bonus": action_bonus,
            "raw_reward": raw_reward,
            "reward_normalised": normalised_reward,
            "reward_running_mean": self._reward_mean,
            "reward_running_std": reward_std,
        }

    def _normalise_reward(self, value: float) -> Tuple[float, float]:
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


def _default_asset_path() -> str:
    from pathlib import Path

    package_root = Path(__file__).resolve().parents[2]
    return str(package_root / "assets" / "tvc_2d.xml")


def make_mjx_batch(env: Tvc2DEnv, batch_size: int) -> Tuple[mjx.Model, mjx.Data]:
    """Creates MJX buffers for batched rollouts that mirror the environment state.

    Args:
        env: Environment instance supplying the MuJoCo model.
        batch_size: Number of parallel simulations to map via ``jax.vmap``.

    Returns:
        Tuple ``(mjx_model, mjx_data)`` representing the batched simulator.
    """

    mjx_model = mjx.put_model(env.model)
    base_data = mjx.make_data(mjx_model)
    qpos = np.repeat(env.data.qpos[None, :], batch_size, axis=0)
    qvel = np.repeat(env.data.qvel[None, :], batch_size, axis=0)
    ctrl = np.repeat(env.data.ctrl[None, :], batch_size, axis=0)
    mjx_data = base_data.replace(qpos=qpos, qvel=qvel, ctrl=ctrl)
    return mjx_model, mjx_data
