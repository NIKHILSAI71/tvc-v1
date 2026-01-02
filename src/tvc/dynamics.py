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
    mass: float = 280.0  # Matches MuJoCo XML total mass (~280kg)
    inertia: Tuple[float, float, float] = (95.0, 95.0, 5.0)  # Matches 2m tall, 0.3m wide rocket (Scale 1:100)
    thrust_max: float = 8540.0  # 1:100 scale of Merlin 1D
    thrust_min: float = 3416.0  # 40% throttle
    arm: float = 1.15  # Matches XML thrust site location relative to CoM (~-1.15m)
    damping: Tuple[float, float, float] = (0.4, 0.2, 0.8)
    gravity: float = 9.81
    tvc_limit: float = 0.14  # ±8 degrees gimbal

    @classmethod
    def from_model(cls, model) -> "RocketParams":
        """Extract parameters automatically from MuJoCo model."""
        import numpy as np
        import mujoco

        # Mass (use subtree mass of vehicle body to capture all attached geoms)
        vehicle_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "vehicle")
        if vehicle_id < 0:
            raise ValueError("Body 'vehicle' not found in model")
        mass = float(model.body_subtreemass[vehicle_id])

        # Inertia
        inertia = model.body_inertia[vehicle_id]
        
        # Thrust (extract from actuator gear)
        thrust_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "thrust_control")
        if thrust_id < 0:
            raise ValueError("Actuator 'thrust_control' not found")
        # Gear is [fx, fy, fz, tx, ty, tz], we want fz (index 2)
        thrust_max = float(model.actuator_gear[thrust_id, 2])
        thrust_min = thrust_max * 0.4  # Assume 40% min throttle

        # TVC Limit
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "tvc_x")
        if joint_id < 0:
            raise ValueError("Joint 'tvc_x' not found")
        tvc_limit = float(model.jnt_range[joint_id, 1])

        # Arm length (distance from CoM to thrust site)
        # Note: We need bind pose positions.
        # vehicle pos is usually origin of body frame.
        # thrust_site pos is relative to body.
        thrust_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "thrust_site")
        vehicle_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "vehicle_cg")
        
        if thrust_site_id >= 0 and vehicle_site_id >= 0:
            # Create data to compute global positions (kinematics)
            data = mujoco.MjData(model)
            mujoco.mj_kinematics(model, data)
            
            # Get global positions
            thrust_pos = data.site_xpos[thrust_site_id]
            cg_pos = data.site_xpos[vehicle_site_id]
            
            # Arm is scalar distance
            arm = float(np.linalg.norm(thrust_pos - cg_pos))
        else:
            raise ValueError("Critical: 'thrust_site' or 'vehicle_cg' sites missing in XML. Cannot calculate lever arm.")

        return cls(
            mass=mass,
            inertia=(float(inertia[0]), float(inertia[1]), float(inertia[2])),
            thrust_max=thrust_max,
            thrust_min=thrust_min,
            arm=arm,
            tvc_limit=tvc_limit,
            damping=(0.4, 0.2, 0.8),  # Keep default damping (not easily extracted)
            gravity=float(abs(model.opt.gravity[2]))
        )


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

    # Thrust calculation: Use full range [0, thrust_max] for maximum control authority
    # thrust_frac ∈ [0, 1] maps to [0, 8540N]
    # Note: Real Merlin 1D has 40% minimum throttle, but for learning we need full range
    # Hover thrust for 256kg rocket = (256 * 9.81) / 8540 ≈ 29.4%
    thrust_mag = thrust_frac * params.thrust_max
    
    # Thrust FORCE acts upward (+Z in body frame)
    # Engine nozzle points down, but thrust force pushes rocket up!
    # Body frame: +Z = nose direction (up along rocket axis)
    # Gimbal angles: gimbal_x = pitch (rotation around X), gimbal_y = yaw (rotation around Y)
    # When gimbal angles are zero, thrust points straight up (+Z)
    # 
    # Torque convention: With engine at [0, 0, -arm]:
    #   - Positive gimbal_y (yaw) → Tx > 0 → Torque around Y (turn left)
    #   - Positive gimbal_x (pitch) → Ty > 0 → Torque around X (pitch up)
    thrust_body = jnp.array([
        thrust_mag * jnp.sin(gimbal_y),      # Lateral thrust (yaw control)
        thrust_mag * jnp.sin(gimbal_x),      # Lateral thrust (pitch control)
        thrust_mag * jnp.cos(gimbal_x) * jnp.cos(gimbal_y),  # Positive = upward thrust!
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

    # Thrust application point (engine is 'arm' meters below center of mass)
    # thrust_offset points from CoM to engine location: [0, 0, -arm] (downward)
    thrust_offset = jnp.array([0.0, 0.0, -params.arm])
    
    # Torque = r × F (cross product of lever arm and thrust force)
    # Gimbal angles deflect thrust laterally, creating torque around CoM
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


def state_to_observation(
    state: jnp.ndarray,
    target_pos: jnp.ndarray | None = None,
    target_vel: jnp.ndarray | None = None,
    target_quat: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Map full state vector to policy observation with target information.
    
    CRITICAL UPDATE: Now includes gimbal state for closed-loop control.
    
    Enhanced observation space for better learning:
    - Current state (position, velocity, orientation, angular velocity)
    - Gimbal state (angles, velocities) - CRITICAL for feedback control
    - Target information (position, velocity, orientation)
    - Error signals (position error, velocity error, orientation alignment)
    - Additional features (distance to target, vertical alignment)
    
    This provides the policy with goal-directed information essential for
    learning stable control towards a target state.
    
    State format (17 elements):
    - [0:3] = position
    - [3:6] = velocity
    - [6:10] = quaternion
    - [10:13] = angular velocity
    - [13:15] = gimbal angles [tvc_x, tvc_y] (NEW)
    - [15:17] = gimbal velocities (NEW)
    """
    pos = state[:3]
    vel = state[3:6]
    quat = state[6:10]
    omega = state[10:13]
    
    # Extract gimbal state if provided
    if state.shape[0] >= 17:
        gimbal_angles = state[13:15]  # [tvc_x, tvc_y]
        gimbal_velocities = state[15:17]
    else:
        # Backwards compatibility: if old state format, use zeros
        gimbal_angles = jnp.zeros(2)
        gimbal_velocities = jnp.zeros(2)
    
    # Current state representation
    R = quaternion_to_rotation_matrix(quat)
    
    # Default targets (hover at current position if not specified)
    if target_pos is None:
        target_pos = jnp.array([0.0, 0.0, 50.0])
    if target_vel is None:
        target_vel = jnp.array([0.0, 0.0, 0.0])
    if target_quat is None:
        target_quat = jnp.array([1.0, 0.0, 0.0, 0.0])  # Upright
    
    # Error signals (CRITICAL for learning goal-directed behavior)
    pos_error = pos - target_pos
    vel_error = vel - target_vel
    distance_to_target = jnp.linalg.norm(pos_error)
    
    # Orientation alignment: dot product of current and target quaternions
    # Close to 1.0 = well aligned, close to 0 = perpendicular, close to -1 = inverted
    orientation_alignment = jnp.abs(jnp.dot(quat, target_quat))
    
    # Vertical alignment: how upright is the rocket (z-axis pointing up)
    # R[:, 2] is the body's z-axis in world frame
    vertical_alignment = R[2, 2]  # Should be close to 1.0 when upright
    
    # Angular velocity magnitude (want this small for stability)
    omega_magnitude = jnp.linalg.norm(omega)
    
    # Comprehensive observation vector with gimbal feedback
    observation = jnp.concatenate([
        pos,                           # Current position (3)
        vel,                           # Current velocity (3)
        R.reshape(-1),                 # Current orientation matrix (9)
        omega,                         # Current angular velocity (3)
        gimbal_angles,                 # CRITICAL: Current gimbal angles [tvc_x, tvc_y] (2)
        gimbal_velocities,             # CRITICAL: Gimbal angular velocities (2)
        target_pos,                    # Target position (3)
        target_vel,                    # Target velocity (3)
        pos_error,                     # Position error vector (3)
        vel_error,                     # Velocity error vector (3)
        jnp.array([distance_to_target]),      # Scalar distance to target (1)
        jnp.array([orientation_alignment]),   # Orientation alignment score (1)
        jnp.array([vertical_alignment]),      # Vertical alignment score (1)
        jnp.array([omega_magnitude]),         # Angular velocity magnitude (1)
    ])
    
    # Total observation size: 3+3+9+3+2+2+3+3+3+3+1+1+1+1 = 38 dimensions (was 32)
    return observation


def hover_state(altitude: float = 8.0) -> jnp.ndarray:
    """Return hovering equilibrium state at given altitude.

    Default altitude: 8m (FIXED: Reduced from 50m for learnable initial training).
    """
    return jnp.array([
        0.0, 0.0, altitude,  # position
        0.0, 0.0, 0.0,  # velocity
        1.0, 0.0, 0.0, 0.0,  # quaternion (identity)
        0.0, 0.0, 0.0,  # angular velocity
    ])
