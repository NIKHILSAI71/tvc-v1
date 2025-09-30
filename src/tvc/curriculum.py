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
    """Build progressive 3D curriculum."""

    # Stage 1: Hover at 8m
    hover_stage = CurriculumStage(
        name="hover_stabilization",
        episodes=200,
        target_position=(0.0, 0.0, 8.0),
        target_orientation=(1.0, 0.0, 0.0, 0.0),  # upright
        target_velocity=(0.0, 0.0, 0.0),
        target_angular_velocity=(0.0, 0.0, 0.0),
        initial_position=(0.0, 0.0, 8.0),
        initial_velocity=(0.0, 0.0, 0.0),
        initial_orientation=(1.0, 0.0, 0.0, 0.0),
        initial_angular_velocity=(0.0, 0.0, 0.0),
        position_tolerance=1.2,
        velocity_tolerance=0.8,
        orientation_tolerance=0.25,
        angular_velocity_tolerance=0.6,
        tolerance_bonus=0.5,
        reward_threshold=0.15,
        success_episodes=4,
        min_episodes=60,
    )

    # Stage 2: Lateral translation (5m offset)
    lateral_stage = CurriculumStage(
        name="lateral_translation",
        episodes=180,
        target_position=(0.0, 0.0, 8.0),
        target_orientation=(1.0, 0.0, 0.0, 0.0),
        target_velocity=(0.0, 0.0, 0.0),
        target_angular_velocity=(0.0, 0.0, 0.0),
        initial_position=(5.0, 0.0, 8.0),
        initial_velocity=(-0.3, 0.0, 0.0),
        initial_orientation=(1.0, 0.0, 0.0, 0.0),
        initial_angular_velocity=(0.0, 0.0, 0.0),
        position_tolerance=1.0,
        velocity_tolerance=0.7,
        orientation_tolerance=0.2,
        angular_velocity_tolerance=0.5,
        tolerance_bonus=0.6,
        reward_threshold=0.2,
        success_episodes=4,
        min_episodes=50,
    )

    # Stage 3: Altitude climb (8m → 12m)
    climb_stage = CurriculumStage(
        name="altitude_climb",
        episodes=160,
        target_position=(0.0, 0.0, 12.0),
        target_orientation=(1.0, 0.0, 0.0, 0.0),
        target_velocity=(0.0, 0.0, 0.0),
        target_angular_velocity=(0.0, 0.0, 0.0),
        initial_position=(0.0, 0.0, 8.0),
        initial_velocity=(0.0, 0.0, 0.5),
        initial_orientation=(1.0, 0.0, 0.0, 0.0),
        initial_angular_velocity=(0.0, 0.0, 0.0),
        position_tolerance=1.5,
        velocity_tolerance=0.8,
        orientation_tolerance=0.2,
        angular_velocity_tolerance=0.5,
        tolerance_bonus=0.55,
        reward_threshold=0.18,
        success_episodes=4,
        min_episodes=45,
    )

    # Stage 4: Controlled descent (12m → 5m)
    descent_stage = CurriculumStage(
        name="controlled_descent",
        episodes=200,
        target_position=(0.0, 0.0, 5.0),
        target_orientation=(1.0, 0.0, 0.0, 0.0),
        target_velocity=(0.0, 0.0, 0.0),
        target_angular_velocity=(0.0, 0.0, 0.0),
        initial_position=(0.0, 0.0, 12.0),
        initial_velocity=(0.0, 0.0, -0.6),
        initial_orientation=(1.0, 0.0, 0.0, 0.0),
        initial_angular_velocity=(0.0, 0.0, 0.0),
        position_tolerance=1.0,
        velocity_tolerance=0.6,
        orientation_tolerance=0.15,
        angular_velocity_tolerance=0.4,
        tolerance_bonus=0.65,
        reward_threshold=0.22,
        success_episodes=5,
        min_episodes=70,
    )

    # Stage 5: Landing approach (5m → 1m)
    landing_stage = CurriculumStage(
        name="landing_approach",
        episodes=220,
        target_position=(0.0, 0.0, 1.0),
        target_orientation=(1.0, 0.0, 0.0, 0.0),
        target_velocity=(0.0, 0.0, 0.0),
        target_angular_velocity=(0.0, 0.0, 0.0),
        initial_position=(0.0, 0.0, 5.0),
        initial_velocity=(0.0, 0.0, -0.4),
        initial_orientation=(1.0, 0.0, 0.0, 0.0),
        initial_angular_velocity=(0.0, 0.0, 0.0),
        position_tolerance=0.6,
        velocity_tolerance=0.4,
        orientation_tolerance=0.1,
        angular_velocity_tolerance=0.3,
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