"""Curriculum definition for 3D TVC training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import jax.numpy as jnp


@dataclass(frozen=True)
class CurriculumStage:
    """Training stage with targets and tolerances."""
    name: str
    episodes: int
    target_position: Tuple[float, float, float]
    target_orientation: Tuple[float, float, float, float]  # quaternion (w, x, y, z)
    target_velocity: Tuple[float, float, float]
    target_angular_velocity: Tuple[float, float, float]
    initial_position: Tuple[float, float, float]
    initial_velocity: Tuple[float, float, float]
    initial_orientation: Tuple[float, float, float, float]
    initial_angular_velocity: Tuple[float, float, float]
    position_tolerance: float
    velocity_tolerance: float
    orientation_tolerance: float
    angular_velocity_tolerance: float
    tolerance_bonus: float
    reward_threshold: float | None = None
    success_episodes: int = 3
    min_episodes: int = 0


def build_curriculum() -> List[CurriculumStage]:
    """Build progressive 3D curriculum with realistic SpaceX-style landing altitudes.

    Simulates a landing burn sequence from 50m hover down to touchdown.
    Realistic velocities based on F9 landing profile (scaled to model).
    """

    # Stage 1: High hover stabilization at 50m (extended for proper learning)
    hover_stage = CurriculumStage(
        name="hover_stabilization",
        episodes=300,  # Increased from 200 for better learning
        target_position=(0.0, 0.0, 50.0),
        target_orientation=(1.0, 0.0, 0.0, 0.0),  # upright
        target_velocity=(0.0, 0.0, 0.0),
        target_angular_velocity=(0.0, 0.0, 0.0),
        initial_position=(0.0, 0.0, 50.0),
        initial_velocity=(0.0, 0.0, 0.0),
        initial_orientation=(1.0, 0.0, 0.0, 0.0),
        initial_angular_velocity=(0.0, 0.0, 0.0),
        position_tolerance=3.0,  # Relaxed from 2.0 for initial learning
        velocity_tolerance=1.5,  # Relaxed from 1.0
        orientation_tolerance=0.2,  # Relaxed from 0.15
        angular_velocity_tolerance=0.5,  # Relaxed from 0.4
        tolerance_bonus=0.5,
        reward_threshold=0.15,
        success_episodes=5,  # Need more consistent success
        min_episodes=100,  # Increased minimum
    )

    # Stage 2: Lateral translation with offset (10m lateral at 50m altitude)
    lateral_stage = CurriculumStage(
        name="lateral_translation",
        episodes=180,
        target_position=(0.0, 0.0, 50.0),
        target_orientation=(1.0, 0.0, 0.0, 0.0),
        target_velocity=(0.0, 0.0, 0.0),
        target_angular_velocity=(0.0, 0.0, 0.0),
        initial_position=(10.0, 0.0, 50.0),
        initial_velocity=(-1.5, 0.0, 0.0),  # ~5.4 km/h lateral velocity
        initial_orientation=(1.0, 0.0, 0.0, 0.0),
        initial_angular_velocity=(0.0, 0.0, 0.0),
        position_tolerance=2.5,
        velocity_tolerance=1.2,
        orientation_tolerance=0.15,
        angular_velocity_tolerance=0.4,
        tolerance_bonus=0.6,
        reward_threshold=0.2,
        success_episodes=4,
        min_episodes=50,
    )

    # Stage 3: Altitude climb (50m → 80m)
    climb_stage = CurriculumStage(
        name="altitude_climb",
        episodes=160,
        target_position=(0.0, 0.0, 80.0),
        target_orientation=(1.0, 0.0, 0.0, 0.0),
        target_velocity=(0.0, 0.0, 0.0),
        target_angular_velocity=(0.0, 0.0, 0.0),
        initial_position=(0.0, 0.0, 50.0),
        initial_velocity=(0.0, 0.0, 2.0),  # 2 m/s climb rate
        initial_orientation=(1.0, 0.0, 0.0, 0.0),
        initial_angular_velocity=(0.0, 0.0, 0.0),
        position_tolerance=3.0,
        velocity_tolerance=1.5,
        orientation_tolerance=0.15,
        angular_velocity_tolerance=0.4,
        tolerance_bonus=0.55,
        reward_threshold=0.18,
        success_episodes=4,
        min_episodes=45,
    )

    # Stage 4: Controlled descent (80m → 20m) - Main landing burn phase
    descent_stage = CurriculumStage(
        name="controlled_descent",
        episodes=200,
        target_position=(0.0, 0.0, 20.0),
        target_orientation=(1.0, 0.0, 0.0, 0.0),
        target_velocity=(0.0, 0.0, 0.0),
        target_angular_velocity=(0.0, 0.0, 0.0),
        initial_position=(0.0, 0.0, 80.0),
        initial_velocity=(0.0, 0.0, -3.0),  # -3 m/s descent rate
        initial_orientation=(1.0, 0.0, 0.0, 0.0),
        initial_angular_velocity=(0.0, 0.0, 0.0),
        position_tolerance=2.0,
        velocity_tolerance=1.0,
        orientation_tolerance=0.12,
        angular_velocity_tolerance=0.3,
        tolerance_bonus=0.65,
        reward_threshold=0.22,
        success_episodes=5,
        min_episodes=70,
    )

    # Stage 5: Final landing approach (20m → 2m) - Precision landing
    landing_stage = CurriculumStage(
        name="landing_approach",
        episodes=220,
        target_position=(0.0, 0.0, 2.0),
        target_orientation=(1.0, 0.0, 0.0, 0.0),
        target_velocity=(0.0, 0.0, -0.5),  # Slow descent -0.5 m/s
        target_angular_velocity=(0.0, 0.0, 0.0),
        initial_position=(0.0, 0.0, 20.0),
        initial_velocity=(0.0, 0.0, -2.0),  # -2 m/s initial descent
        initial_orientation=(1.0, 0.0, 0.0, 0.0),
        initial_angular_velocity=(0.0, 0.0, 0.0),
        position_tolerance=1.0,
        velocity_tolerance=0.8,
        orientation_tolerance=0.08,
        angular_velocity_tolerance=0.2,
        tolerance_bonus=0.7,
        reward_threshold=0.25,
        success_episodes=5,
        min_episodes=80,
    )

    return [hover_stage, lateral_stage, climb_stage, descent_stage, landing_stage]


def select_stage(curriculum: List[CurriculumStage], episode: int) -> CurriculumStage:
    """Return active curriculum stage for episode."""
    counter = 0
    for stage in curriculum:
        counter += stage.episodes
        if episode < counter:
            return stage
    return curriculum[-1]