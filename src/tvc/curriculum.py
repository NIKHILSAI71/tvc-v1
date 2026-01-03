"""Reverse Curriculum for accelerating landing learning."""

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
    """Build REVERSE curriculum: Land first, then fly higher."""

    # Stage 1: The "Last Meter" (Touchdown)
    # Start extremely close to the ground (5m).
    # Goal: Just kill velocity and upright yourself.
    # High success rate expected immediately.
    touchdown_stage = CurriculumStage(
        name="touchdown_practice",
        episodes=300,
        target_position=(0.0, 0.0, 1.0), # Land on pad (1m buffer?)
        target_orientation=(1.0, 0.0, 0.0, 0.0),
        target_velocity=(0.0, 0.0, -0.5), # Soft touch
        target_angular_velocity=(0.0, 0.0, 0.0),
        
        initial_position=(0.0, 0.0, 5.0), # 5m altitude
        initial_velocity=(0.0, 0.0, -1.0), # Already descending
        initial_orientation=(1.0, 0.0, 0.0, 0.0),
        initial_angular_velocity=(0.0, 0.0, 0.0),
        
        position_tolerance=2.0,
        velocity_tolerance=1.0,
        orientation_tolerance=0.3, # ~15 deg
        angular_velocity_tolerance=0.5,
        tolerance_bonus=0.5,
        success_episodes=5,
        min_episodes=50
    )

    # Stage 2: Intermediate Hop (NEW - bridges difficulty gap)
    # Start at 10m. Gentler transition from 5m to 15m.
    # Research: Curriculum stages should increase difficulty by 30-50%, not 200%
    intermediate_hop_stage = CurriculumStage(
        name="intermediate_hop",
        episodes=200,
        target_position=(0.0, 0.0, 1.0),
        target_orientation=(1.0, 0.0, 0.0, 0.0),
        target_velocity=(0.0, 0.0, -0.5),
        target_angular_velocity=(0.0, 0.0, 0.0),
        
        initial_position=(0.0, 0.0, 10.0),  # 10m, not 15m (100% increase from 5m)
        initial_velocity=(0.0, 0.0, -1.5),   # Gentler descent than short_hop
        initial_orientation=(1.0, 0.0, 0.0, 0.0),
        initial_angular_velocity=(0.0, 0.0, 0.0),
        
        position_tolerance=1.8,  # Slightly more forgiving than short_hop
        velocity_tolerance=0.9,
        orientation_tolerance=0.25,
        angular_velocity_tolerance=0.45,
        tolerance_bonus=0.55,
        success_episodes=5,
        min_episodes=40
    )

    # Stage 3: Short Hop Landing (was Stage 2)
    # Start at 15m. Must control descent.
    short_hop_stage = CurriculumStage(
        name="short_hop",
        episodes=300,
        target_position=(0.0, 0.0, 1.0),
        target_orientation=(1.0, 0.0, 0.0, 0.0),
        target_velocity=(0.0, 0.0, -0.5),
        target_angular_velocity=(0.0, 0.0, 0.0),
        
        initial_position=(0.0, 0.0, 15.0),
        initial_velocity=(0.0, 0.0, -2.0),
        initial_orientation=(1.0, 0.0, 0.0, 0.0),
        initial_angular_velocity=(0.0, 0.0, 0.0),
        
        position_tolerance=1.5,
        velocity_tolerance=0.8,
        orientation_tolerance=0.2,
        angular_velocity_tolerance=0.4,
        tolerance_bonus=0.6,
        success_episodes=5,
        min_episodes=50
    )

    # Stage 3: High Altitude Descent
    # Start at 40m. Requires significant thrust management.
    high_descent_stage = CurriculumStage(
        name="high_descent",
        episodes=400,
        target_position=(0.0, 0.0, 1.0),
        target_orientation=(1.0, 0.0, 0.0, 0.0),
        target_velocity=(0.0, 0.0, -0.5),
        target_angular_velocity=(0.0, 0.0, 0.0),
        
        initial_position=(0.0, 0.0, 40.0),
        initial_velocity=(0.0, 0.0, -5.0), # Falling fast!
        initial_orientation=(1.0, 0.0, 0.0, 0.0),
        initial_angular_velocity=(0.0, 0.0, 0.0),
        
        position_tolerance=1.5,
        velocity_tolerance=0.8,
        orientation_tolerance=0.15,
        angular_velocity_tolerance=0.3,
        tolerance_bonus=0.8,
        success_episodes=5,
        min_episodes=100
    )
    
    # Stage 4: Expert (Disturbances)
    # Start high, lateral offset, random noise.
    expert_stage = CurriculumStage(
        name="expert_landing",
        episodes=500,
        target_position=(0.0, 0.0, 1.0),
        target_orientation=(1.0, 0.0, 0.0, 0.0),
        target_velocity=(0.0, 0.0, -0.2),
        target_angular_velocity=(0.0, 0.0, 0.0),
        
        initial_position=(8.0, 0.0, 40.0), # Lateral offset
        initial_velocity=(0.0, 0.0, -2.0),
        initial_orientation=(1.0, 0.0, 0.0, 0.0),
        initial_angular_velocity=(0.0, 0.0, 0.0),
        
        position_tolerance=0.5,
        velocity_tolerance=0.5,
        orientation_tolerance=0.1,
        angular_velocity_tolerance=0.2,
        tolerance_bonus=1.0,
        success_episodes=10,
        min_episodes=100
    )

    return [touchdown_stage, intermediate_hop_stage, short_hop_stage, high_descent_stage, expert_stage]


def select_stage(curriculum: List[CurriculumStage], episode: int) -> CurriculumStage:
    """Return active curriculum stage for episode."""
    counter = 0
    for stage in curriculum:
        counter += stage.episodes
        if episode < counter:
            return stage
    return curriculum[-1]
