"""Top-level package for advanced 2D TVC control research tooling."""

from importlib import import_module
from typing import TYPE_CHECKING

from .runtime import ensure_jax_runtime

ensure_jax_runtime()

if TYPE_CHECKING:  # pragma: no cover - import-time side effects only
    from .curriculum import build_curriculum
    from .env import Tvc2DEnv, TvcGymnasiumEnv, make_mjx_batch
    from .mpc import compute_tvc_mpc_action
    from .policies import build_policy_network, evaluate_policy
    from .training import save_policy_checkpoints, train_controller
else:
    _env = import_module(".env", __name__)
    Tvc2DEnv = _env.Tvc2DEnv
    TvcGymnasiumEnv = _env.TvcGymnasiumEnv
    make_mjx_batch = _env.make_mjx_batch

    _mpc = import_module(".mpc", __name__)
    compute_tvc_mpc_action = _mpc.compute_tvc_mpc_action

    _policies = import_module(".policies", __name__)
    build_policy_network = _policies.build_policy_network
    evaluate_policy = _policies.evaluate_policy

    _training = import_module(".training", __name__)
    train_controller = _training.train_controller
    save_policy_checkpoints = _training.save_policy_checkpoints

    _curriculum = import_module(".curriculum", __name__)
    build_curriculum = _curriculum.build_curriculum

__all__ = [
    "Tvc2DEnv",
    "TvcGymnasiumEnv",
    "make_mjx_batch",
    "compute_tvc_mpc_action",
    "build_policy_network",
    "evaluate_policy",
    "train_controller",
    "save_policy_checkpoints",
    "build_curriculum",
]
