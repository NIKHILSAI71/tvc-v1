"""Environment utilities for the 2D thrust-vector-control research platform.

This module exposes a lightweight MuJoCo wrapper that keeps the vehicle within a
2D plane while enabling accurate rigid-body dynamics. It also provides helpers
for interacting with the MJX GPU-accelerated backend so that large training
batches can be simulated efficiently when curriculum schedules demand many
parallel scenarios.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple, TYPE_CHECKING

import math

import mujoco as _mujoco
import numpy as np
from mujoco import mjx

from .dynamics import RocketParams

if TYPE_CHECKING:  # pragma: no cover - typing aid
    from .curriculum import CurriculumStage

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
        self._stage_config = {
            "target_altitude": 4.0,
            "target_pitch": 0.0,
            "target_vertical_velocity": 0.0,
            "target_lateral": 0.0,
            "phase": "hover",
            "initial_altitude": None,
            "initial_vertical_velocity": 0.0,
            "initial_pitch": 0.0,
            "initial_pitch_rate": 0.0,
            "initial_lateral": 0.0,
            "initial_lateral_velocity": 0.0,
            "pitch_tolerance": 0.12,
            "pitch_rate_tolerance": 0.45,
            "altitude_tolerance": 0.5,
            "vertical_velocity_tolerance": 0.6,
            "lateral_tolerance": 0.6,
            "lateral_velocity_tolerance": 0.6,
            "tolerance_bonus": 0.45,
        }
        mujoco.mj_forward(self.model, self.data)

    # ---------------------------------------------------------------------
    # Stage configuration helpers
    # ---------------------------------------------------------------------

    def configure_stage(self, stage: "CurriculumStage") -> None:
        """Applies curriculum-specific targets and initial conditions."""

        stage_dict: Dict[str, Any] = {
            "target_altitude": float(getattr(stage, "target_altitude", 4.0)),
            "target_pitch": float(getattr(stage, "target_pitch", 0.0)),
            "target_vertical_velocity": float(getattr(stage, "target_vertical_velocity", 0.0)),
            "target_lateral": float(getattr(stage, "target_lateral", 0.0)),
            "phase": getattr(stage, "phase", "hover"),
            "initial_altitude": getattr(stage, "initial_altitude", None),
            "initial_vertical_velocity": float(getattr(stage, "initial_vertical_velocity", 0.0)),
            "initial_pitch": float(getattr(stage, "initial_pitch", 0.0)),
            "initial_pitch_rate": float(getattr(stage, "initial_pitch_rate", 0.0)),
            "initial_lateral": float(getattr(stage, "initial_lateral", 0.0)),
            "initial_lateral_velocity": float(getattr(stage, "initial_lateral_velocity", 0.0)),
            "pitch_tolerance": float(getattr(stage, "pitch_tolerance", 0.12)),
            "pitch_rate_tolerance": float(getattr(stage, "pitch_rate_tolerance", 0.45)),
            "altitude_tolerance": float(getattr(stage, "altitude_tolerance", 0.5)),
            "vertical_velocity_tolerance": float(getattr(stage, "vertical_velocity_tolerance", 0.6)),
            "lateral_tolerance": float(getattr(stage, "lateral_tolerance", 0.6)),
            "lateral_velocity_tolerance": float(getattr(stage, "lateral_velocity_tolerance", 0.6)),
            "tolerance_bonus": float(getattr(stage, "tolerance_bonus", 0.45)),
        }
        self._stage_config.update(stage_dict)

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
        config = self._stage_config
        base_altitude = config.get("initial_altitude")
        base_altitude = 3.5 if base_altitude is None else float(base_altitude)
        self.data.qpos[2] = base_altitude
        self.data.qvel[:] = 0.0

        base_vertical_velocity = float(config.get("initial_vertical_velocity", 0.0))
        self.data.qvel[2] = base_vertical_velocity

        # Inject bounded disturbances to mimic pre-launch uncertainties.
        base_pitch = float(config.get("initial_pitch", 0.0))
        angle_noise = self._rng.normal(0.0, 0.02 * disturbance_scale)
        mujoco.mju_axisAngle2Quat(
            self.data.qpos[3:7], np.array([1.0, 0.0, 0.0]), base_pitch + angle_noise
        )
        base_pitch_rate = float(config.get("initial_pitch_rate", 0.0))
        self.data.qvel[3] = self._rng.normal(0.0, 0.5 * disturbance_scale)
        self.data.qvel[4] = base_pitch_rate + self._rng.normal(0.0, 0.5 * disturbance_scale)
        self.data.qvel[5] = self._rng.normal(0.0, 0.5 * disturbance_scale)

        # Random lateral offsets.
        base_lateral = float(config.get("initial_lateral", 0.0))
        base_lateral_velocity = float(config.get("initial_lateral_velocity", 0.0))
        self.data.qpos[0] = base_lateral + self._rng.normal(0.0, 0.3 * disturbance_scale)
        self.data.qvel[0] = base_lateral_velocity + self._rng.normal(0.0, 0.5 * disturbance_scale)
        self.data.qvel[2] += self._rng.normal(0.0, 0.4 * disturbance_scale)

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
        config = self._stage_config
        pitch_tolerance = max(float(config.get("pitch_tolerance", 0.12)), 1e-6)
        pitch_rate_tol = max(float(config.get("pitch_rate_tolerance", 0.45)), 1e-6)
        lateral_tolerance = max(float(config.get("lateral_tolerance", 0.6)), 1e-6)
        lateral_vel_tol = max(float(config.get("lateral_velocity_tolerance", 0.6)), 1e-6)
        altitude_tolerance = max(float(config.get("altitude_tolerance", 0.5)), 1e-6)
        vertical_vel_tol = max(float(config.get("vertical_velocity_tolerance", 0.6)), 1e-6)

        target_pitch = float(config.get("target_pitch", 0.0))
        target_altitude = float(config.get("target_altitude", 4.0))
        target_lateral = float(config.get("target_lateral", 0.0))
        target_vertical_vel = float(config.get("target_vertical_velocity", 0.0))

        lateral = float(self.data.qpos[0])
        lateral_error = lateral - target_lateral
        lateral_vel = float(self.data.qvel[0])
        altitude = float(self.data.qpos[2])
        altitude_error = target_altitude - altitude
        vertical_vel = float(self.data.qvel[2])
        vertical_vel_error = vertical_vel - target_vertical_vel
        pitch_error = pitch - target_pitch

        return np.array(
            [
                thrust,
                pitch_error / pitch_tolerance,
                pitch_rate / pitch_rate_tol,
                lateral_error / lateral_tolerance,
                lateral_vel / lateral_vel_tol,
                altitude_error / altitude_tolerance,
                vertical_vel_error / vertical_vel_tol,
            ],
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
        config = self._stage_config
        target_pitch = float(config.get("target_pitch", 0.0))
        target_altitude = float(config.get("target_altitude", 4.0))
        target_vertical_velocity = float(config.get("target_vertical_velocity", 0.0))
        target_lateral = float(config.get("target_lateral", 0.0))

        pitch = self._pitch()
        pitch_error = pitch - target_pitch
        pitch_rate = self._pitch_rate()
        lateral = float(self.data.qpos[0]) - target_lateral
        lateral_vel = float(self.data.qvel[0])
        altitude = float(self.data.qpos[2])
        altitude_error = target_altitude - altitude
        vertical_vel_raw = float(self.data.qvel[2])
        vertical_vel_error = vertical_vel_raw - target_vertical_velocity
        jerk = float(np.linalg.norm(self.data.qacc[:3]))
        ctrl_jitter = float(np.linalg.norm(np.diff(self.data.ctrl)))
        ctrl_magnitude = float(np.linalg.norm(self.data.ctrl[1:3]))

        pitch_scale = max(float(config.get("pitch_tolerance", 0.12)), 1e-6)
        pitch_rate_scale = max(float(config.get("pitch_rate_tolerance", 0.45)), 1e-6)
        lateral_scale = max(float(config.get("lateral_tolerance", 0.6)), 1e-6)
        lateral_vel_scale = max(float(config.get("lateral_velocity_tolerance", 0.6)), 1e-6)
        altitude_scale = max(float(config.get("altitude_tolerance", 0.5)), 1e-6)
        vertical_scale = max(float(config.get("vertical_velocity_tolerance", 0.6)), 1e-6)

        def _exp_penalty(value: float, scale: float) -> float:
            return 1.0 - math.exp(-abs(value) / scale)

        penalty = (
            0.42 * _exp_penalty(pitch_error, pitch_scale)
            + 0.18 * _exp_penalty(pitch_rate, pitch_rate_scale)
            + 0.22 * _exp_penalty(lateral, lateral_scale)
            + 0.16 * _exp_penalty(lateral_vel, lateral_vel_scale)
            + 0.32 * _exp_penalty(altitude_error, altitude_scale)
            + 0.24 * _exp_penalty(vertical_vel_error, vertical_scale)
            + 0.05 * _exp_penalty(jerk, 18.0)
            + 0.04 * _exp_penalty(ctrl_jitter, 0.18)
            + 0.03 * _exp_penalty(ctrl_magnitude, 0.2)
        )

        stability_bonus = float(
            0.35
            * math.exp(-0.5 * ((pitch_error / max(pitch_scale * 0.8, 1e-6)) ** 2 + (lateral / max(lateral_scale * 0.8, 1e-6)) ** 2))
        )
        hover_bonus = float(
            0.25
            * math.exp(
                -0.5
                * (
                    (altitude_error / max(altitude_scale * 0.75, 1e-6)) ** 2
                    + (vertical_vel_error / max(vertical_scale * 0.8, 1e-6)) ** 2
                )
            )
        )
        attitude_bonus = float(0.22 * math.exp(-0.5 * (pitch_rate / max(pitch_rate_scale * 0.8, 1e-6)) ** 2))
        action_bonus = float(0.12 * math.exp(-0.5 * (ctrl_magnitude / 0.18) ** 2))

        within_pitch = abs(pitch_error) <= pitch_scale
        within_altitude = abs(altitude_error) <= altitude_scale
        within_lateral = abs(lateral) <= lateral_scale
        within_vertical = abs(vertical_vel_error) <= vertical_scale
        tolerance_bonus = float(config.get("tolerance_bonus", 0.45)) if (
            within_pitch and within_altitude and within_lateral and within_vertical
        ) else 0.0

        phase_bonus = 0.0
        phase = config.get("phase", "hover")
        if phase == "launch":
            ascent_velocity = max(0.0, vertical_vel_raw - target_vertical_velocity)
            phase_bonus = float(0.12 * math.tanh(ascent_velocity / max(vertical_scale, 1e-6)))
        elif phase == "landing":
            descent_error = abs(vertical_vel_error)
            phase_bonus = float(0.1 * math.exp(-descent_error / max(vertical_scale, 1e-6)))
        elif phase == "attitude":
            phase_bonus = float(0.08 * math.exp(-abs(pitch_error) / max(pitch_scale * 0.6, 1e-6)))

        raw_reward = (
            1.2
            - penalty
            + stability_bonus
            + hover_bonus
            + attitude_bonus
            + action_bonus
            + phase_bonus
            + tolerance_bonus
        )
        raw_reward = float(np.clip(raw_reward, -5.0, 3.0))
        normalised_reward, reward_std = self._normalise_reward(raw_reward)

        return normalised_reward, {
            "pitch": pitch,
            "pitch_error": pitch_error,
            "pitch_rate": pitch_rate,
            "lateral": lateral,
            "lateral_vel": lateral_vel,
            "altitude": altitude,
            "altitude_error": altitude_error,
            "vertical_velocity": vertical_vel_raw,
            "vertical_velocity_error": vertical_vel_error,
            "jerk": jerk,
            "ctrl_jitter": ctrl_jitter,
            "ctrl_magnitude": ctrl_magnitude,
            "reward_penalty": penalty,
            "stability_bonus": stability_bonus,
            "hover_bonus": hover_bonus,
            "attitude_bonus": attitude_bonus,
            "action_bonus": action_bonus,
            "phase_bonus": phase_bonus,
            "tolerance_bonus": tolerance_bonus,
            "raw_reward": raw_reward,
            "reward_normalised": normalised_reward,
            "reward_running_mean": self._reward_mean,
            "reward_running_std": reward_std,
            "target_pitch": target_pitch,
            "target_altitude": target_altitude,
            "target_vertical_velocity": target_vertical_velocity,
            "pitch_tolerance": pitch_scale,
            "altitude_tolerance": altitude_scale,
            "vertical_velocity_tolerance": vertical_scale,
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
