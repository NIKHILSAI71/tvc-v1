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


def build_curriculum() -> List[CurriculumStage]:
    """Builds a progressive curriculum covering increasingly harsh scenarios."""

    hover_target = jnp.array([0.0, 4.0, 0.0, 0.0, 0.0, 0.0])
    translational_target = hover_target.at[0].set(1.5)
    gust_target = hover_target.at[3].set(-1.0)

    return [
        CurriculumStage(
            "pad_hover",
            0.1,
            hover_target,
            episodes=500,
            reward_threshold=-160.0,
            success_episodes=2,
            min_episodes=120,
        ),
        CurriculumStage(
            "lateral_reject",
            0.3,
            translational_target,
            episodes=1200,
            reward_threshold=-140.0,
            success_episodes=3,
            min_episodes=200,
        ),
        CurriculumStage(
            "wind_gust",
            0.6,
            gust_target,
            episodes=1800,
            reward_threshold=-125.0,
            success_episodes=4,
            min_episodes=320,
        ),
    ]


def select_stage(curriculum: List[CurriculumStage], episode: int) -> CurriculumStage:
    """Returns the active curriculum stage for the given episode index."""

    counter = 0
    for stage in curriculum:
        counter += stage.episodes
        if episode < counter:
            return stage
    return curriculum[-1]
