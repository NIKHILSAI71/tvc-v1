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
            0.05,
            hover_target,
            episodes=450,
            reward_threshold=-80.0,
            success_episodes=3,
            min_episodes=60,
        ),
        CurriculumStage(
            "lateral_reject",
            0.25,
            translational_target,
            episodes=1000,
            reward_threshold=-110.0,
            success_episodes=3,
            min_episodes=160,
        ),
        CurriculumStage(
            "wind_gust",
            0.5,
            gust_target,
            episodes=1600,
            reward_threshold=-100.0,
            success_episodes=4,
            min_episodes=260,
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
