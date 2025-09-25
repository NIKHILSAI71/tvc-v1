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
) -> Tuple[jnp.ndarray, Dict[str, float]]:
    """Optimises a sequence of TVC offsets and returns the first action.

    Args:
        state: Current rocket state ``[x, z, vx, vz, theta, omega]``.
        reference: Desired terminal state vector with identical layout to ``state``.
        params: Physical parameter bundle describing the rocket.
        config: MPC hyperparameters controlling horizon, weights, and optimiser.

    Returns:
        Tuple ``(action, diagnostics)`` where ``action`` is the two-element TVC command.
    """

    horizon = config.horizon
    controls = jnp.zeros((horizon, 2))
    controls_flat = controls.reshape(-1)

    def loss_fn(control_flat: jnp.ndarray) -> jnp.ndarray:
        shaped_controls = control_flat.reshape((horizon, 2))
        return _cost_function(state, shaped_controls, reference, config, params)

    value_and_grad = jax.value_and_grad(loss_fn)

    def opt_step(carry, _):
        control_flat, opt_state = carry
        loss, grads = value_and_grad(control_flat)
        updates, opt_state = optimiser.update(grads, opt_state)
        control_flat = optax.apply_updates(control_flat, updates)
        control_flat = jnp.asarray(control_flat)
        control_matrix = jnp.reshape(control_flat, (horizon, 2))
        control_matrix = _project_controls(control_matrix, config.control_limit)
        control_flat = jnp.reshape(control_matrix, (-1,))
        return (control_flat, opt_state), loss

    optimiser = optax.adam(config.learning_rate)
    opt_state = optimiser.init(controls_flat)
    (control_flat, opt_state), losses = jax.lax.scan(opt_step, (controls_flat, opt_state), None, length=config.iterations)

    controls = control_flat.reshape((horizon, 2))
    action = controls[0]
    diagnostics = {
        "mpc_loss": float(losses[-1]),
        "optim_iterations": float(config.iterations),
    }
    return action, diagnostics
