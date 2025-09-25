"""Top-level package for advanced 2D TVC control research tooling."""

from .runtime import ensure_jax_runtime

ensure_jax_runtime()

from .env import Tvc2DEnv, make_mjx_batch
from .mpc import compute_tvc_mpc_action
from .policies import build_policy_network, evaluate_policy
from .training import train_controller
from .curriculum import build_curriculum

__all__ = [
    "Tvc2DEnv",
    "make_mjx_batch",
    "compute_tvc_mpc_action",
    "build_policy_network",
    "evaluate_policy",
    "train_controller",
    "build_curriculum",
]
