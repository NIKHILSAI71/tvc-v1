"""JAX-based 3D rocket dynamics for thrust vector control."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class RocketParams:
    """Physical parameters for the 3D rocket model.

    Based on 1:100 scale SpaceX Falcon 9 first stage (realistic model rocket scale):
    - Full scale: 25,600 kg, 70m tall, 854 kN thrust
    - Model scale: 256 kg, 0.7m tall, 8540 N thrust
    - Maintains real thrust-to-weight ratio: ~3.4
    - Real Merlin 1D throttle: 40-100%
    - Real TVC range: ±8-10 degrees (~0.14-0.17 rad)

    Args:
        mass: Vehicle mass (kg) - scaled from F9 empty mass.
        inertia: Diagonal inertia tensor (Ixx, Iyy, Izz) (kg·m²) - realistic ratios.
        thrust_max: Maximum thrust magnitude (N) - 100% throttle.
        thrust_min: Minimum thrust magnitude (N) - 40% throttle (real Merlin limit).
        arm: Lever arm between nozzle pivot and centre of mass (m).
        damping: Aerodynamic damping (linear_xy, linear_z, angular) - reduced for altitude.
        gravity: Gravitational acceleration (m/s²) - Earth standard.
        tvc_limit: Maximum TVC gimbal angle (radians) - real F9 limit ~±8°.
    """
    mass: float = 256.0  # 1:100 scale of F9 empty mass (25,600 kg)
    inertia: Tuple[float, float, float] = (680.0, 680.0, 45.0)  # Realistic cylindrical body ratios
    thrust_max: float = 8540.0  # 1:100 scale of Merlin 1D (854 kN) → T/W = 3.4
    thrust_min: float = 3416.0  # 40% throttle (real Merlin minimum)
    arm: float = 2.0  # Engine to CoM distance
    damping: Tuple[float, float, float] = (0.4, 0.2, 0.8)  # Reduced for high-altitude flight
    gravity: float = 9.81  # Earth surface gravity
    tvc_limit: float = 0.14  # ±8 degrees gimbal (real F9 spec)


def quaternion_multiply(q1: jnp.ndarray, q2: jnp.ndarray) -> jnp.ndarray:
    """Multiply two quaternions in (w, x, y, z) convention."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return jnp.array([w, x, y, z])


def quaternion_to_rotation_matrix(q: jnp.ndarray) -> jnp.ndarray:
    """Convert quaternion to rotation matrix."""
    w, x, y, z = q
    norm = jnp.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return jnp.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def rocket_step(
    state: jnp.ndarray,
    control: jnp.ndarray,
    dt: float,
    params: RocketParams
) -> jnp.ndarray:
    """Integrate 6DOF rocket dynamics forward by one step.

    Args:
        state: [x, y, z, vx, vy, vz, qw, qx, qy, qz, wx, wy, wz]
        control: [gimbal_x, gimbal_y, thrust_frac]
        dt: Integration timestep (seconds)
        params: Physical parameters

    Returns:
        Next state vector
    """
    pos = state[:3]
    vel = state[3:6]
    quat = state[6:10]
    omega = state[10:13]

    gimbal_x, gimbal_y, thrust_frac = control
    gimbal_x = jnp.clip(gimbal_x, -params.tvc_limit, params.tvc_limit)
    gimbal_y = jnp.clip(gimbal_y, -params.tvc_limit, params.tvc_limit)
    thrust_frac = jnp.clip(thrust_frac, 0.0, 1.0)

    thrust_mag = params.thrust_min + thrust_frac * (params.thrust_max - params.thrust_min)
    thrust_body = jnp.array([
        thrust_mag * jnp.sin(gimbal_y),
        -thrust_mag * jnp.sin(gimbal_x),
        -thrust_mag * jnp.cos(gimbal_x) * jnp.cos(gimbal_y),
    ])

    R = quaternion_to_rotation_matrix(quat)
    thrust_world = R @ thrust_body

    gravity_force = jnp.array([0.0, 0.0, -params.mass * params.gravity])
    drag_force = -jnp.array([
        params.damping[0] * vel[0],
        params.damping[0] * vel[1],
        params.damping[1] * vel[2],
    ])
    total_force = thrust_world + gravity_force + drag_force
    accel = total_force / params.mass

    thrust_offset = jnp.array([0.0, 0.0, -params.arm])
    tvc_torque_body = jnp.cross(thrust_offset, thrust_body)
    angular_damping = -params.damping[2] * omega
    total_torque_body = tvc_torque_body + angular_damping
    inertia = jnp.array(params.inertia)
    alpha = total_torque_body / inertia

    vel_next = vel + dt * accel
    pos_next = pos + dt * vel_next
    omega_next = omega + dt * alpha

    omega_quat = jnp.array([0.0, omega_next[0], omega_next[1], omega_next[2]])
    quat_dot = 0.5 * quaternion_multiply(quat, omega_quat)
    quat_next = quat + dt * quat_dot
    quat_next = quat_next / jnp.linalg.norm(quat_next)

    return jnp.concatenate([pos_next, vel_next, quat_next, omega_next])


def linearise_dynamics(
    state: jnp.ndarray,
    control: jnp.ndarray,
    dt: float,
    params: RocketParams,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Compute discrete-time Jacobians (A, B) for MPC."""
    def transition(s, u):
        return rocket_step(s, u, dt, params)

    A = jax.jacfwd(transition, argnums=0)(state, control)
    B = jax.jacfwd(transition, argnums=1)(state, control)
    return A, B


def simulate_rollout(
    state: jnp.ndarray,
    controls: jnp.ndarray,
    dt: float,
    params: RocketParams,
) -> jnp.ndarray:
    """Simulate dynamics over horizon using jax.lax.scan."""
    def scan_fn(carry, u):
        nxt = rocket_step(carry, u, dt, params)
        return nxt, nxt

    _, states = jax.lax.scan(scan_fn, state, controls)
    return states


def state_to_observation(state: jnp.ndarray) -> jnp.ndarray:
    """Map full state vector to policy observation."""
    pos = state[:3]
    vel = state[3:6]
    quat = state[6:10]
    omega = state[10:13]
    R = quaternion_to_rotation_matrix(quat)
    return jnp.concatenate([pos, vel, R.reshape(-1), omega])


def hover_state(altitude: float = 50.0) -> jnp.ndarray:
    """Return hovering equilibrium state at given altitude.

    Default altitude: 50m (realistic landing scenario start point).
    """
    return jnp.array([
        0.0, 0.0, altitude,  # position
        0.0, 0.0, 0.0,  # velocity
        1.0, 0.0, 0.0, 0.0,  # quaternion (identity)
        0.0, 0.0, 0.0,  # angular velocity
    ])