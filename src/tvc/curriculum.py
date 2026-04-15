"""Reverse Curriculum for accelerating landing learning."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
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


def build_curriculum(config_path: Path | None = None) -> List[CurriculumStage]:
    """Build REVERSE curriculum from configuration: Hover first, then land from increasing heights.
    
    CRITICAL: Each stage must be MASTERED before advancing.
    Stage 0 (Hover) teaches basic stabilization - foundation for all later skills.
    """
    if config_path is None:
        config_path = Path(__file__).parent / "curriculum.json"
        
    with open(config_path, "r") as f:
        stages_data = json.load(f)
        
    stages = []
    for data in stages_data:
        # Convert lists to tuples for the CurriculumStage initialization
        stage = CurriculumStage(
            name=data["name"],
            episodes=data["episodes"],
            target_position=tuple(data["target_position"]),
            target_orientation=tuple(data["target_orientation"]),
            target_velocity=tuple(data["target_velocity"]),
            target_angular_velocity=tuple(data["target_angular_velocity"]),
            initial_position=tuple(data["initial_position"]),
            initial_velocity=tuple(data["initial_velocity"]),
            initial_orientation=tuple(data["initial_orientation"]),
            initial_angular_velocity=tuple(data["initial_angular_velocity"]),
            position_tolerance=data["position_tolerance"],
            velocity_tolerance=data["velocity_tolerance"],
            orientation_tolerance=data["orientation_tolerance"],
            angular_velocity_tolerance=data["angular_velocity_tolerance"],
            tolerance_bonus=data["tolerance_bonus"],
            success_episodes=data["success_episodes"],
            min_episodes=data["min_episodes"],
        )
        stages.append(stage)
        
    return stages


def select_stage(curriculum: List[CurriculumStage], episode: int) -> CurriculumStage:
    """Return active curriculum stage for episode."""
    counter = 0
    for stage in curriculum:
        counter += stage.episodes
        if episode < counter:
            return stage
    return curriculum[-1]
