"""Curriculum definition for multi-scenario TVC training."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import jax.numpy as jnp


@dataclass(frozen=True)
class CurriculumStage:
    """Encapsulates stochastic disturbances, targets, and advancement rules."""

    name: str
    disturbance_scale: float
    target_state: jnp.ndarray
    episodes: int
    reward_threshold: float | None = None
    success_episodes: int = 3
    min_episodes: int = 0
    target_pitch: float = 0.0
    target_altitude: float = 4.0
    target_vertical_velocity: float = 0.0
    target_lateral: float = 0.0
    phase: str = "hover"
    initial_altitude: float | None = None
    initial_vertical_velocity: float = 0.0
    initial_pitch: float = 0.0
    initial_pitch_rate: float = 0.0
    initial_lateral: float = 0.0
    initial_lateral_velocity: float = 0.0
    pitch_tolerance: float = 0.12
    pitch_rate_tolerance: float = 0.45
    altitude_tolerance: float = 0.5
    vertical_velocity_tolerance: float = 0.6
    lateral_tolerance: float = 0.6
    lateral_velocity_tolerance: float = 0.6
    tolerance_bonus: float = 0.45


def build_curriculum() -> List[CurriculumStage]:
    """Builds a progressive curriculum covering increasingly harsh scenarios."""

    def _state(x: float, z: float, vx: float = 0.0, vz: float = 0.0, theta: float = 0.0, omega: float = 0.0) -> jnp.ndarray:
        return jnp.array([x, z, vx, vz, theta, omega])

    launch_altitude = 6.5
    launch_stage = CurriculumStage(
        name="launch_ascent",
        disturbance_scale=0.06,
        target_state=_state(0.0, launch_altitude, 0.0, 0.0, 0.0, 0.0),
        episodes=160,
        reward_threshold=0.2,
        success_episodes=4,
        min_episodes=40,
        target_pitch=0.0,
        target_altitude=launch_altitude,
        target_vertical_velocity=0.0,
        phase="launch",
        initial_altitude=1.2,
        initial_vertical_velocity=0.2,
        initial_pitch=0.0,
        initial_pitch_rate=0.0,
        initial_lateral=0.0,
        initial_lateral_velocity=0.0,
        pitch_tolerance=0.18,
        pitch_rate_tolerance=0.55,
        altitude_tolerance=0.9,
        vertical_velocity_tolerance=0.8,
        lateral_tolerance=0.8,
        lateral_velocity_tolerance=0.75,
        tolerance_bonus=0.35,
    )

    attitude_angle = 0.12  # ~6.9 degrees
    angle_stage = CurriculumStage(
        name="angle_hold",
        disturbance_scale=0.08,
        target_state=_state(0.0, launch_altitude, 0.0, 0.0, attitude_angle, 0.0),
        episodes=180,
        reward_threshold=0.18,
        success_episodes=4,
        min_episodes=50,
        target_pitch=attitude_angle,
        target_altitude=launch_altitude,
        target_vertical_velocity=0.0,
        phase="attitude",
        initial_altitude=6.0,
        initial_vertical_velocity=0.0,
        initial_pitch=0.02,
        initial_pitch_rate=0.0,
        initial_lateral=0.0,
        initial_lateral_velocity=0.0,
        pitch_tolerance=0.08,
        pitch_rate_tolerance=0.4,
        altitude_tolerance=0.7,
        vertical_velocity_tolerance=0.6,
        lateral_tolerance=0.55,
        lateral_velocity_tolerance=0.55,
        tolerance_bonus=0.5,
    )

    hover_altitude = 4.0
    hover_stage = CurriculumStage(
        name="pad_hover",
        disturbance_scale=0.12,
        target_state=_state(0.0, hover_altitude, 0.0, 0.0, 0.0, 0.0),
        episodes=200,
        reward_threshold=0.25,
        success_episodes=5,
        min_episodes=60,
        target_pitch=0.0,
        target_altitude=hover_altitude,
        phase="hover",
        initial_altitude=hover_altitude,
        initial_vertical_velocity=0.0,
        initial_pitch=0.0,
        pitch_tolerance=0.06,
        pitch_rate_tolerance=0.32,
        altitude_tolerance=0.45,
        vertical_velocity_tolerance=0.4,
        lateral_tolerance=0.45,
        lateral_velocity_tolerance=0.45,
        tolerance_bonus=0.55,
    )

    landing_stage = CurriculumStage(
        name="powered_descent",
        disturbance_scale=0.18,
        target_state=_state(0.0, 0.8, 0.0, 0.0, 0.0, 0.0),
        episodes=220,
        reward_threshold=0.12,
        success_episodes=5,
        min_episodes=80,
        target_pitch=0.0,
        target_altitude=0.8,
        target_vertical_velocity=0.0,
        phase="landing",
        initial_altitude=7.0,
        initial_vertical_velocity=-0.4,
        initial_pitch=0.02,
        initial_pitch_rate=0.0,
        pitch_tolerance=0.07,
        pitch_rate_tolerance=0.35,
        altitude_tolerance=0.6,
        vertical_velocity_tolerance=0.45,
        lateral_tolerance=0.5,
        lateral_velocity_tolerance=0.5,
        tolerance_bonus=0.5,
    )

    touchdown_stage = CurriculumStage(
        name="landing_touchdown",
        disturbance_scale=0.12,
        target_state=_state(0.0, 0.6, 0.0, 0.0, 0.0, 0.0),
        episodes=260,
        reward_threshold=0.1,
        success_episodes=5,
        min_episodes=80,
        target_pitch=0.0,
        target_altitude=0.6,
        target_vertical_velocity=0.0,
        phase="landing",
        initial_altitude=1.4,
        initial_vertical_velocity=-0.2,
        initial_pitch=0.0,
        pitch_tolerance=0.05,
        pitch_rate_tolerance=0.3,
        altitude_tolerance=0.35,
        vertical_velocity_tolerance=0.35,
        lateral_tolerance=0.4,
        lateral_velocity_tolerance=0.4,
        tolerance_bonus=0.6,
    )

    return [launch_stage, angle_stage, hover_stage, landing_stage, touchdown_stage]


def select_stage(curriculum: List[CurriculumStage], episode: int) -> CurriculumStage:
    """Returns the active curriculum stage for the given episode index."""

    counter = 0
    for stage in curriculum:
        counter += stage.episodes
        if episode < counter:
            return stage
    return curriculum[-1]
