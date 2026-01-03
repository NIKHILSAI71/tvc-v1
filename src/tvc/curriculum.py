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
    """Build REVERSE curriculum: Hover first, then land from increasing heights.
    
    CRITICAL: Each stage must be MASTERED before advancing.
    Stage 0 (Hover) teaches basic stabilization - foundation for all later skills.
    """

    # ============================================================
    # STAGE 0: HOVER MASTERY - Foundation skill
    # Goal: Learn to maintain position and orientation at low altitude
    # This prevents the model from learning to "fall with style"
    # ============================================================
    hover_stage = CurriculumStage(
        name="3m Hover",
        episodes=250,
        target_position=(0.0, 0.0, 3.0),  # Hover at 3m
        target_orientation=(1.0, 0.0, 0.0, 0.0),
        target_velocity=(0.0, 0.0, 0.0),  # Zero velocity (true hover)
        target_angular_velocity=(0.0, 0.0, 0.0),
        
        initial_position=(0.0, 0.0, 3.5),  # Start slightly above target
        initial_velocity=(0.0, 0.0, -0.3),  # Gentle initial descent
        initial_orientation=(1.0, 0.0, 0.0, 0.0),
        initial_angular_velocity=(0.0, 0.0, 0.0),
        
        position_tolerance=0.8,   # Must hold position tightly
        velocity_tolerance=0.4,   # Near-zero velocity required
        orientation_tolerance=0.2,  # ~11 degrees max tilt
        angular_velocity_tolerance=0.3,
        tolerance_bonus=0.5,
        success_episodes=10,  # Must succeed multiple times consistently
        min_episodes=100      # Minimum training before advancement
    )

    # ============================================================
    # STAGE 1: Easy Descent - 5 meters
    # Goal: Learn controlled descent and soft touchdown
    # Tighter tolerances than before to ensure real mastery
    # ============================================================
    touchdown_stage = CurriculumStage(
        name="5m Easy",
        episodes=350,
        target_position=(0.0, 0.0, 1.0),  # Land on pad
        target_orientation=(1.0, 0.0, 0.0, 0.0),
        target_velocity=(0.0, 0.0, -0.3),  # Softer touch than before
        target_angular_velocity=(0.0, 0.0, 0.0),
        
        initial_position=(0.0, 0.0, 5.0),  # 5m altitude
        initial_velocity=(0.0, 0.0, -0.8),  # Gentler initial descent
        initial_orientation=(1.0, 0.0, 0.0, 0.0),
        initial_angular_velocity=(0.0, 0.0, 0.0),
        
        position_tolerance=1.5,   # Tighter than before (was 2.0)
        velocity_tolerance=0.6,   # Must land softly (was 1.0)
        orientation_tolerance=0.2,  # ~11 deg (was 0.3)
        angular_velocity_tolerance=0.4,  # Less wobble (was 0.5)
        tolerance_bonus=0.6,
        success_episodes=8,   # More consistent success needed (was 5)
        min_episodes=120      # More training required (was 50)
    )

    # ============================================================
    # STAGE 2: Medium Descent - 10 meters
    # Bridges the difficulty gap with stricter requirements
    # ============================================================
    intermediate_hop_stage = CurriculumStage(
        name="10m Medium",
        episodes=300,
        target_position=(0.0, 0.0, 1.0),
        target_orientation=(1.0, 0.0, 0.0, 0.0),
        target_velocity=(0.0, 0.0, -0.3),
        target_angular_velocity=(0.0, 0.0, 0.0),
        
        initial_position=(0.0, 0.0, 10.0),
        initial_velocity=(0.0, 0.0, -1.2),  # Moderate descent
        initial_orientation=(1.0, 0.0, 0.0, 0.0),
        initial_angular_velocity=(0.0, 0.0, 0.0),
        
        position_tolerance=1.2,
        velocity_tolerance=0.5,
        orientation_tolerance=0.18,
        angular_velocity_tolerance=0.35,
        tolerance_bonus=0.65,
        success_episodes=8,
        min_episodes=100
    )

    # ============================================================
    # STAGE 3: Hard Descent - 15 meters
    # Must control faster descent, requires good timing
    # ============================================================
    short_hop_stage = CurriculumStage(
        name="15m Hard",
        episodes=350,
        target_position=(0.0, 0.0, 1.0),
        target_orientation=(1.0, 0.0, 0.0, 0.0),
        target_velocity=(0.0, 0.0, -0.3),
        target_angular_velocity=(0.0, 0.0, 0.0),
        
        initial_position=(0.0, 0.0, 15.0),
        initial_velocity=(0.0, 0.0, -1.8),
        initial_orientation=(1.0, 0.0, 0.0, 0.0),
        initial_angular_velocity=(0.0, 0.0, 0.0),
        
        position_tolerance=1.0,
        velocity_tolerance=0.5,
        orientation_tolerance=0.15,
        angular_velocity_tolerance=0.3,
        tolerance_bonus=0.7,
        success_episodes=8,
        min_episodes=120
    )

    # ============================================================
    # STAGE 4: High Altitude Pro - 40 meters
    # Fast falling, requires excellent control and timing
    # ============================================================
    high_descent_stage = CurriculumStage(
        name="40m Pro",
        episodes=400,
        target_position=(0.0, 0.0, 1.0),
        target_orientation=(1.0, 0.0, 0.0, 0.0),
        target_velocity=(0.0, 0.0, -0.3),
        target_angular_velocity=(0.0, 0.0, 0.0),
        
        initial_position=(0.0, 0.0, 40.0),
        initial_velocity=(0.0, 0.0, -4.0),  # Fast descent
        initial_orientation=(1.0, 0.0, 0.0, 0.0),
        initial_angular_velocity=(0.0, 0.0, 0.0),
        
        position_tolerance=1.0,
        velocity_tolerance=0.4,
        orientation_tolerance=0.12,
        angular_velocity_tolerance=0.25,
        tolerance_bonus=0.85,
        success_episodes=10,
        min_episodes=150
    )
    
    # ============================================================
    # STAGE 5: Expert with Lateral Offset - Hardest
    # Start high with lateral offset - requires angled descent
    # ============================================================
    expert_stage = CurriculumStage(
        name="40m Expert",
        episodes=500,
        target_position=(0.0, 0.0, 1.0),
        target_orientation=(1.0, 0.0, 0.0, 0.0),
        target_velocity=(0.0, 0.0, -0.2),
        target_angular_velocity=(0.0, 0.0, 0.0),
        
        initial_position=(8.0, 0.0, 40.0),  # 8m lateral offset
        initial_velocity=(0.0, 0.0, -2.0),
        initial_orientation=(1.0, 0.0, 0.0, 0.0),
        initial_angular_velocity=(0.0, 0.0, 0.0),
        
        position_tolerance=0.5,
        velocity_tolerance=0.3,
        orientation_tolerance=0.1,
        angular_velocity_tolerance=0.2,
        tolerance_bonus=1.0,
        success_episodes=15,  # Very consistent success needed
        min_episodes=150
    )

    return [hover_stage, touchdown_stage, intermediate_hop_stage, short_hop_stage, high_descent_stage, expert_stage]


def select_stage(curriculum: List[CurriculumStage], episode: int) -> CurriculumStage:
    """Return active curriculum stage for episode."""
    counter = 0
    for stage in curriculum:
        counter += stage.episodes
        if episode < counter:
            return stage
    return curriculum[-1]
