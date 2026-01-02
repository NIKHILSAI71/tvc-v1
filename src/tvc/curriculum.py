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

    # Stage 1: LOW hover stabilization at 8m (CRITICAL FIX: Reduced from 50m for learnable initial training)
    # Starting at 8m makes initial stabilization much more achievable
    hover_stage = CurriculumStage(
        name="hover_stabilization",
        episodes=300,  # Sufficient episodes for learning
        target_position=(0.0, 0.0, 8.0),  # 8m hover height
        target_orientation=(1.0, 0.0, 0.0, 0.0),  # upright
        target_velocity=(0.0, 0.0, 0.0),
        target_angular_velocity=(0.0, 0.0, 0.0),
        initial_position=(0.0, 0.0, 8.0),  # Start at target
        initial_velocity=(0.0, 0.0, 0.0),
        initial_orientation=(1.0, 0.0, 0.0, 0.0),
        initial_angular_velocity=(0.0, 0.0, 0.0),
        position_tolerance=1.5,  # RELAXED: 1.5m for initial learning (was 1.0)
        velocity_tolerance=1.0,  # RELAXED: 1.0 m/s (was 0.8)
        orientation_tolerance=0.20,  # RELAXED: 0.20 rad ≈ 11.5° (was 0.15)
        angular_velocity_tolerance=0.4,  # RELAXED: 0.4 rad/s (was 0.3)
        tolerance_bonus=0.5,
        reward_threshold=0.15,
        success_episodes=3,  # CRITICAL: Reduced to 3 consistent successes (was 5)
        min_episodes=50,  # Reduced minimum to allow faster progression
    )

    # Stage 2: Lateral translation with offset (5m lateral at 8m altitude)
    # Reduced from 10m offset at 50m to 5m offset at 8m for progressive difficulty
    lateral_stage = CurriculumStage(
        name="lateral_translation",
        episodes=180,
        target_position=(0.0, 0.0, 8.0),  # 8m altitude
        target_orientation=(1.0, 0.0, 0.0, 0.0),
        target_velocity=(0.0, 0.0, 0.0),
        target_angular_velocity=(0.0, 0.0, 0.0),
        initial_position=(5.0, 0.0, 8.0),  # 5m lateral offset (was 10m)
        initial_velocity=(-1.0, 0.0, 0.0),  # Reduced from -1.5 m/s
        initial_orientation=(1.0, 0.0, 0.0, 0.0),
        initial_angular_velocity=(0.0, 0.0, 0.0),
        position_tolerance=1.5,  # Tightened from 2.5m
        velocity_tolerance=1.0,  # Tightened from 1.2
        orientation_tolerance=0.15,
        angular_velocity_tolerance=0.4,
        tolerance_bonus=0.6,
        reward_threshold=0.2,
        success_episodes=4,
        min_episodes=50,
    )

    # Stage 3: Altitude climb (8m → 30m)
    # Progressive altitude increase from 8m to 30m (was 50m → 80m)
    climb_stage = CurriculumStage(
        name="altitude_climb",
        episodes=160,
        target_position=(0.0, 0.0, 30.0),  # 30m target
        target_orientation=(1.0, 0.0, 0.0, 0.0),
        target_velocity=(0.0, 0.0, 0.0),
        target_angular_velocity=(0.0, 0.0, 0.0),
        initial_position=(0.0, 0.0, 8.0),  # Start from 8m
        initial_velocity=(0.0, 0.0, 1.5),  # Reduced from 2.0 m/s
        initial_orientation=(1.0, 0.0, 0.0, 0.0),
        initial_angular_velocity=(0.0, 0.0, 0.0),
        position_tolerance=2.5,  # Tightened from 3.0
        velocity_tolerance=1.2,  # Tightened from 1.5
        orientation_tolerance=0.15,
        angular_velocity_tolerance=0.4,
        tolerance_bonus=0.55,
        reward_threshold=0.18,
        success_episodes=4,
        min_episodes=45,
    )

    # Stage 4: Controlled descent (30m → 8m) - Main landing burn phase
    # Adjusted to work with lower altitude progression
    descent_stage = CurriculumStage(
        name="controlled_descent",
        episodes=200,
        target_position=(0.0, 0.0, 8.0),  # Return to 8m hover
        target_orientation=(1.0, 0.0, 0.0, 0.0),
        target_velocity=(0.0, 0.0, 0.0),
        target_angular_velocity=(0.0, 0.0, 0.0),
        initial_position=(0.0, 0.0, 30.0),  # Start from 30m
        initial_velocity=(0.0, 0.0, -2.0),  # Reduced from -3.0 m/s
        initial_orientation=(1.0, 0.0, 0.0, 0.0),
        initial_angular_velocity=(0.0, 0.0, 0.0),
        position_tolerance=1.5,  # Tightened from 2.0
        velocity_tolerance=0.8,  # Tightened from 1.0
        orientation_tolerance=0.12,
        angular_velocity_tolerance=0.3,
        tolerance_bonus=0.65,
        reward_threshold=0.22,
        success_episodes=5,
        min_episodes=70,
    )

    # Stage 5: Final landing approach (8m → 2m) - Precision landing
    # Adjusted to work with lower altitude progression
    landing_stage = CurriculumStage(
        name="landing_approach",
        episodes=220,
        target_position=(0.0, 0.0, 2.0),
        target_orientation=(1.0, 0.0, 0.0, 0.0),
        target_velocity=(0.0, 0.0, -0.3),  # Slower descent -0.3 m/s (was -0.5)
        target_angular_velocity=(0.0, 0.0, 0.0),
        initial_position=(0.0, 0.0, 8.0),  # Start from 8m (was 20m)
        initial_velocity=(0.0, 0.0, -1.0),  # Reduced from -2.0 m/s
        initial_orientation=(1.0, 0.0, 0.0, 0.0),
        initial_angular_velocity=(0.0, 0.0, 0.0),
        position_tolerance=0.8,  # Tightened from 1.0
        velocity_tolerance=0.6,  # Tightened from 0.8
        orientation_tolerance=0.08,
        angular_velocity_tolerance=0.2,
        tolerance_bonus=0.7,
        reward_threshold=0.25,
        success_episodes=5,
        min_episodes=80,
    )

    # Stage 6: EXPERT (Real World Chaos) - Unbeatable Adaptation
    # Full disturbances enabled by env randomization, strict tolerances
    expert_stage = CurriculumStage(
        name="expert_storm_landing",
        episodes=500,  # Long mastery phase
        target_position=(0.0, 0.0, 1.0),  # Precision touchdown
        target_orientation=(1.0, 0.0, 0.0, 0.0),
        target_velocity=(0.0, 0.0, -0.2),  # Gentle touch
        target_angular_velocity=(0.0, 0.0, 0.0),
        initial_position=(0.0, 0.0, 20.0),  # Start higher
        initial_velocity=(0.0, 0.0, -2.0),
        initial_orientation=(1.0, 0.0, 0.0, 0.0),
        initial_angular_velocity=(0.0, 0.0, 0.0),
        position_tolerance=0.4,  # Sub-meter precision
        velocity_tolerance=0.3,
        orientation_tolerance=0.05,  # Razor sharp < 6 degrees
        angular_velocity_tolerance=0.1,
        tolerance_bonus=1.0,  # Huge bonus for perfection
        reward_threshold=0.28,
        success_episodes=10,  # Must prove it consistently
        min_episodes=100,
    )

    return [hover_stage, lateral_stage, climb_stage, descent_stage, landing_stage, expert_stage]


def select_stage(curriculum: List[CurriculumStage], episode: int) -> CurriculumStage:
    """Return active curriculum stage for episode."""
    counter = 0
    for stage in curriculum:
        counter += stage.episodes
        if episode < counter:
            return stage
    return curriculum[-1]
