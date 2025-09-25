"""Environment utilities for the 2D thrust-vector-control research platform.

This module exposes a lightweight MuJoCo wrapper that keeps the vehicle within a
2D plane while enabling accurate rigid-body dynamics. It also provides helpers
for interacting with the MJX GPU-accelerated backend so that large training
batches can be simulated efficiently when curriculum schedules demand many
parallel scenarios.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import mujoco
import numpy as np
from mujoco import mjx


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
        ctrl_limit: float = 0.3,
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
        return np.array([thrust, pitch, pitch_rate], dtype=np.float32)

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
        jerk = float(np.linalg.norm(self.data.qacc[:3]))
        ctrl_smooth = float(np.linalg.norm(np.diff(self.data.ctrl)))

        penalty = 4.0 * pitch**2 + 0.5 * pitch_rate**2 + 0.3 * lateral**2 + 0.1 * lateral_vel**2
        penalty += 0.05 * jerk + 0.02 * ctrl_smooth
        reward = float(-penalty)
        return reward, {
            "pitch_error": pitch,
            "pitch_rate": pitch_rate,
            "lateral": lateral,
            "lateral_vel": lateral_vel,
            "jerk": jerk,
        }


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
