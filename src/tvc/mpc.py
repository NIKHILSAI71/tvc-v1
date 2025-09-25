"""Gradient-based MPC solver for planar thrust vector control."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import jax
import jax.numpy as jnp
import optax

from .dynamics import RocketParams, simulate_rollout


@dataclass(frozen=True)
class MpcConfig:
    """Configuration bundle for the MPC optimisation routine."""

    horizon: int = 24
    dt: float = 0.05
    iterations: int = 40
    learning_rate: float = 0.08
    control_limit: float = 0.28
    position_weight: float = 40.0
    attitude_weight: float = 120.0
    velocity_weight: float = 6.0
    control_weight: float = 0.5
    terminal_weight: float = 2.5
    tolerance: float = 1e-4
    grad_clip: float | None = 2.5


def _cost_function(
    state: jnp.ndarray,
    controls: jnp.ndarray,
    reference: jnp.ndarray,
    config: MpcConfig,
    params: RocketParams,
) -> jnp.ndarray:
    states = simulate_rollout(state, controls, config.dt, params)
    # Append the initial state for convenience when penalising velocities.
    states_full = jnp.vstack([state[None, :], states])

    position_error = states_full[:, :2] - reference[:2]
    velocity_error = states_full[:, 2:4] - reference[2:4]
    attitude_error = states_full[:, 4:] - reference[4:]

    pos_cost = config.position_weight * jnp.sum(jnp.square(position_error))
    vel_cost = config.velocity_weight * jnp.sum(jnp.square(velocity_error))
    att_cost = config.attitude_weight * jnp.sum(jnp.square(attitude_error))
    ctrl_cost = config.control_weight * jnp.sum(jnp.square(controls[1:] - controls[:-1]))
    term_cost = config.terminal_weight * jnp.sum(jnp.square(states_full[-1] - reference))
    return pos_cost + vel_cost + att_cost + ctrl_cost + term_cost


@jax.jit
def _project_controls(controls: jnp.ndarray, limit: float) -> jnp.ndarray:
    return jnp.clip(controls, -limit, limit)


def compute_tvc_mpc_action(
    state: jnp.ndarray,
    reference: jnp.ndarray,
    params: RocketParams,
    config: MpcConfig = MpcConfig(),
    warm_start: jnp.ndarray | None = None,
) -> Tuple[jnp.ndarray, Dict[str, float], jnp.ndarray]:
    """Optimises a sequence of TVC offsets and returns the first action.

    Args:
        state: Current rocket state ``[x, z, vx, vz, theta, omega]``.
        reference: Desired terminal state vector with identical layout to ``state``.
        params: Physical parameter bundle describing the rocket.
        config: MPC hyperparameters controlling horizon, weights, and optimiser.

    Returns:
        Tuple ``(action, diagnostics, plan)`` where ``action`` is the two-element TVC
        command and ``plan`` is the optimised horizon of control offsets.
    """

    horizon = config.horizon
    controls = jnp.zeros((horizon, 2))
    if warm_start is not None:
        warm_start = jnp.asarray(warm_start, dtype=jnp.float32)
        if warm_start.shape == (horizon, 2):
            controls = _project_controls(warm_start, config.control_limit)
    controls_flat = controls.reshape(-1)

    def loss_fn(control_flat: jnp.ndarray) -> jnp.ndarray:
        shaped_controls = control_flat.reshape((horizon, 2))
        return _cost_function(state, shaped_controls, reference, config, params)

    initial_loss = float(loss_fn(controls_flat))
    value_and_grad = jax.value_and_grad(loss_fn)

    def opt_step(carry, _):
        control_flat, opt_state = carry
        loss, grads = value_and_grad(control_flat)
        grad_norm = jnp.linalg.norm(grads)
        if config.grad_clip is not None and config.grad_clip > 0:
            max_norm = jnp.asarray(config.grad_clip, dtype=grads.dtype)
            scale = jnp.minimum(1.0, max_norm / (grad_norm + 1e-8))
            grads = grads * scale
        updates, opt_state = optimiser.update(grads, opt_state)
        control_flat = optax.apply_updates(control_flat, updates)
        control_flat = jnp.asarray(control_flat)
        control_matrix = jnp.reshape(control_flat, (horizon, 2))
        control_matrix = _project_controls(control_matrix, config.control_limit)
        control_flat = jnp.reshape(control_matrix, (-1,))
        return (control_flat, opt_state), (loss, grad_norm)

    optimiser = optax.adam(config.learning_rate)
    opt_state = optimiser.init(controls_flat)
    (control_flat, opt_state), opt_stats = jax.lax.scan(opt_step, (controls_flat, opt_state), None, length=config.iterations)

    losses = opt_stats[0]
    grad_norms = opt_stats[1]
    controls = control_flat.reshape((horizon, 2))
    action = controls[0]
    final_loss = float(losses[-1]) if losses.size else initial_loss
    loss_deltas = jnp.abs(losses[1:] - losses[:-1]) if losses.size > 1 else jnp.zeros((0,), dtype=losses.dtype)
    indices = jnp.arange(1, config.iterations, dtype=jnp.int32)
    default_iters = jnp.array(config.iterations, dtype=jnp.int32)
    if config.tolerance > 0 and loss_deltas.size > 0:
        converged = jnp.where(loss_deltas < config.tolerance, indices, default_iters)
        effective_iters = int(jnp.min(converged).item())
        if effective_iters == config.iterations:
            effective_iters = config.iterations
    else:
        effective_iters = config.iterations
    saturation_ratio = float(jnp.mean((jnp.abs(controls) >= (config.control_limit - 1e-4)).astype(jnp.float32)))
    grad_norm_final = float(grad_norms[-1]) if grad_norms.size else 0.0
    diagnostics = {
        "mpc_loss": final_loss,
        "mpc_initial_loss": initial_loss,
        "mpc_grad_norm": grad_norm_final,
        "mpc_iterations": float(effective_iters),
        "mpc_saturation": saturation_ratio,
    }
    return action, diagnostics, controls
