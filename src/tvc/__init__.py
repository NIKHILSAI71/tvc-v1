"""Production 3D Thrust Vector Control package."""

from .curriculum import CurriculumStage, build_curriculum, select_stage
from .dynamics import RocketParams, hover_state, rocket_step, simulate_rollout
from .env import StepResult, TvcEnv
from .policies import PolicyConfig, PolicyFunctions, build_policy_network, mutate_parameters

__version__ = "1.0.0"

__all__ = [
    "TvcEnv",
    "StepResult",
    "RocketParams",
    "build_policy_network",
    "rocket_step",
    "simulate_rollout",
    "hover_state",
    "mutate_parameters",
]
