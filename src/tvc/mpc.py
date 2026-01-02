"""Gradient-based MPC solver for 3D thrust vector control."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import jax
import jax.numpy as jnp
from jax import Array

from .dynamics import RocketParams, simulate_rollout


@dataclass(frozen=True)
class MpcConfig:
    """Configuration for 3D MPC optimization."""
    horizon: int = 20  # Increased from 12 (0.48s) to 20 (0.8s) for better foresight
    dt: float = 0.04
    iterations: int = 30
    learning_rate: float = 0.03
    gimbal_limit: float = 0.14
    thrust_min: float = 0.2
    thrust_max: float = 1.0
    tolerance: float = 1e-4
    grad_clip: float | None = 2.0
    position_weight: float = 20.0
    velocity_weight: float = 10.0
    orientation_weight: float = 40.0  # Increased from 35.0 for lock-in alignment
    angular_velocity_weight: float = 8.0
    control_weight: float = 2.0
    control_smoothness_weight: float = 5.0  # Increased from 4.0 to reduce actuator jitter
    thrust_efficiency_weight: float = 0.5
    terminal_weight: float = 15.0  # Increased from 12.0 for precise landing


def _quaternion_distance(q1: jnp.ndarray, q2: jnp.ndarray) -> jnp.ndarray:
    """Compute angular distance between quaternions."""
    dot_product = jnp.abs(jnp.dot(q1, q2))
    dot_product = jnp.clip(dot_product, 0.0, 1.0)
    return 2.0 * jnp.arccos(dot_product)


def _project_controls(controls: Array, config: MpcConfig) -> Array:
    """Project controls to valid ranges."""
    gimbal = jnp.clip(controls[:, :2], -config.gimbal_limit, config.gimbal_limit)
    thrust = jnp.clip(controls[:, 2:3], config.thrust_min, config.thrust_max)
    return jnp.concatenate([gimbal, thrust], axis=1)


def _cost_function(
    state: jnp.ndarray,
    controls: jnp.ndarray,
    reference: jnp.ndarray,
    config: MpcConfig,
    params: RocketParams,
) -> jnp.ndarray:
    """Compute trajectory cost."""
    states = simulate_rollout(state, controls, config.dt, params)
    states_full = jnp.vstack([state[None, :], states])

    positions = states_full[:, 0:3]
    velocities = states_full[:, 3:6]
    quaternions = states_full[:, 6:10]
    ang_velocities = states_full[:, 10:13]

    ref_pos = reference[0:3]
    ref_vel = reference[3:6]
    ref_quat = reference[6:10]
    ref_omega = reference[10:13]

    pos_cost = config.position_weight * jnp.sum(jnp.square(positions - ref_pos[None, :]))
    vel_cost = config.velocity_weight * jnp.sum(jnp.square(velocities - ref_vel[None, :]))

    orient_cost = 0.0
    for q in quaternions:
        orient_cost += config.orientation_weight * _quaternion_distance(q, ref_quat) ** 2

    omega_cost = config.angular_velocity_weight * jnp.sum(jnp.square(ang_velocities - ref_omega[None, :]))
    ctrl_cost = config.control_weight * jnp.sum(jnp.square(controls))

    if controls.shape[0] > 1:
        smoothness = controls[1:] - controls[:-1]
        smoothness_cost = config.control_smoothness_weight * jnp.sum(jnp.square(smoothness))
    else:
        smoothness_cost = 0.0

    thrust = controls[:, 2]
    efficiency_cost = config.thrust_efficiency_weight * jnp.sum(jnp.square(thrust))

    final_pos_error = positions[-1] - ref_pos
    final_vel_error = velocities[-1] - ref_vel
    final_quat_dist = _quaternion_distance(quaternions[-1], ref_quat)
    final_omega_error = ang_velocities[-1] - ref_omega
    terminal_cost = config.terminal_weight * (
        jnp.sum(jnp.square(final_pos_error))
        + jnp.sum(jnp.square(final_vel_error))
        + final_quat_dist**2
        + jnp.sum(jnp.square(final_omega_error))
    )

    return (
        pos_cost
        + vel_cost
        + orient_cost
        + omega_cost
        + ctrl_cost
        + smoothness_cost
        + efficiency_cost
        + terminal_cost
    )


def compute_mpc_action(
    state: Array,
    reference: Array,
    params: RocketParams,
    config: MpcConfig = MpcConfig(),
    warm_start: Array | None = None,
) -> Tuple[Array, Dict[str, float], Array]:
    """Optimize 3D TVC + thrust controls and return the first action.

    Args:
        state: Current state [x, y, z, vx, vy, vz, qw, qx, qy, qz, wx, wy, wz]
        reference: Target state (same format)
        params: Rocket physical parameters
        config: MPC configuration
        warm_start: Optional warm-start control sequence

    Returns:
        action: First control [gimbal_x, gimbal_y, thrust_frac]
        diagnostics: Cost and convergence metrics
        plan: Full optimized control sequence
    """
    horizon = int(config.horizon)

    if warm_start is not None:
        warm_start = jnp.asarray(warm_start, dtype=jnp.float32)
        if warm_start.shape != (horizon, 3):
            raise ValueError(f"warm_start must have shape {(horizon, 3)}, got {warm_start.shape}")
        controls = _project_controls(warm_start, config)
    else:
        hover_thrust = (params.mass * params.gravity) / params.thrust_max
        hover_thrust = jnp.clip(hover_thrust, config.thrust_min, config.thrust_max)
        controls = jnp.zeros((horizon, 3), dtype=jnp.float32)
        controls = controls.at[:, 2].set(hover_thrust)
        controls = _project_controls(controls, config)

    controls_flat = jnp.reshape(controls, (-1,))

    def loss_fn(
        control_flat: Array,
        loss_state: Array,
        loss_reference: Array,
        loss_config: MpcConfig,
        loss_params: RocketParams,
    ) -> Array:
        ctrl = jnp.reshape(control_flat, (horizon, 3))
        ctrl = _project_controls(ctrl, loss_config)
        return _cost_function(loss_state, ctrl, loss_reference, loss_config, loss_params)

    value_and_grad = jax.jit(jax.value_and_grad(loss_fn), static_argnums=(3, 4))
    grad_fn = jax.jit(jax.grad(loss_fn), static_argnums=(3, 4))
    control_flat = controls_flat
    loss_history = []

    for _ in range(config.iterations):
        loss, grads = value_and_grad(control_flat, state, reference, config, params)
        grad_norm = jnp.linalg.norm(grads)
        if config.grad_clip is not None and config.grad_clip > 0:
            scale = jnp.minimum(1.0, config.grad_clip / (grad_norm + 1e-8))
            grads = grads * scale
        control_flat = control_flat - config.learning_rate * grads
        control_matrix = jnp.reshape(jnp.asarray(control_flat), (horizon, 3))
        control_matrix = _project_controls(control_matrix, config)
        control_flat = jnp.reshape(control_matrix, (-1,))
        loss_history.append(float(loss))
        if config.tolerance > 0 and len(loss_history) > 1:
            if abs(loss_history[-1] - loss_history[-2]) < config.tolerance:
                break

    controls = jnp.reshape(control_flat, (horizon, 3))
    action = controls[0]
    final_loss = float(loss_fn(control_flat, state, reference, config, params))
    initial_loss = loss_history[0] if loss_history else final_loss
    best_loss = min(loss_history) if loss_history else final_loss
    grad_norm_final = float(jnp.linalg.norm(grad_fn(control_flat, state, reference, config, params)))

    gimbal_saturation = jnp.mean(jnp.abs(controls[:, :2]) >= (config.gimbal_limit - 1e-4))
    thrust_at_min = jnp.mean(controls[:, 2] <= (config.thrust_min + 1e-4))
    thrust_at_max = jnp.mean(controls[:, 2] >= (config.thrust_max - 1e-4))

    diagnostics = {
        "cost": final_loss,
        "cost_initial": initial_loss,
        "cost_best": best_loss,
        "grad_norm": float(grad_norm_final),
        "iterations": float(len(loss_history)),
        "gimbal_saturation": float(gimbal_saturation),
        "thrust_at_min": float(thrust_at_min),
        "thrust_at_max": float(thrust_at_max),
        "mean_thrust": float(jnp.mean(controls[:, 2])),
        "mean_gimbal_magnitude": float(jnp.mean(jnp.linalg.norm(controls[:, :2], axis=1))),
    }

    return action, diagnostics, controls
