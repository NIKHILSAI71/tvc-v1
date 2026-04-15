import os
import sys

import jax.numpy as jnp
import numpy as np
import pytest

# Ensure src is in the path
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from tvc.dynamics import (
    quaternion_multiply,
    quaternion_to_rotation_matrix,
)

def test_quaternion_multiply_identity():
    # Identity quaternion
    q1 = jnp.array([1.0, 0.0, 0.0, 0.0])
    q2 = jnp.array([1.0, 0.0, 0.0, 0.0])
    q_out = quaternion_multiply(q1, q2)
    np.testing.assert_allclose(q_out, [1.0, 0.0, 0.0, 0.0], atol=1e-6)

def test_quaternion_multiply_rotations():
    # q1 = 90 deg rot around x-axis
    q1 = jnp.array([np.sqrt(0.5), np.sqrt(0.5), 0.0, 0.0])
    # q2 = 90 deg rot around y-axis
    q2 = jnp.array([np.sqrt(0.5), 0.0, np.sqrt(0.5), 0.0])

    # Expected q3 = q1 * q2
    # w = w1*w2 - x1*x2 - y1*y2 - z1*z2 = 0.5
    # x = w1*x2 + x1*w2 + y1*z2 - z1*y2 = 0.5
    # y = w1*y2 - x1*z2 + y1*w2 + z1*x2 = 0.5
    # z = w1*z2 + x1*y2 - y1*x2 + z1*w2 = 0.5
    q_out = quaternion_multiply(q1, q2)
    np.testing.assert_allclose(q_out, [0.5, 0.5, 0.5, 0.5], atol=1e-6)

def test_quaternion_to_rotation_matrix_identity():
    q = jnp.array([1.0, 0.0, 0.0, 0.0])
    R = quaternion_to_rotation_matrix(q)
    R_expected = np.eye(3)
    np.testing.assert_allclose(R, R_expected, atol=1e-6)

def test_quaternion_to_rotation_matrix_x_rotation():
    # 90 degrees around x
    q = jnp.array([np.sqrt(0.5), np.sqrt(0.5), 0.0, 0.0])
    R = quaternion_to_rotation_matrix(q)
    R_expected = np.array([
        [1.0,  0.0,  0.0],
        [0.0,  0.0, -1.0],
        [0.0,  1.0,  0.0]
    ])
    np.testing.assert_allclose(R, R_expected, atol=1e-6)

from tvc.dynamics import rocket_step, RocketParams, hover_state

def test_rocket_step_zero_control_freefall():
    params = RocketParams(mass=100.0, gravity=9.81, thrust_max=1000.0, thrust_min=0.0)
    # Start at origin, zero velocity, identity quat, zero angular vel
    state = jnp.zeros(13)
    state = state.at[6].set(1.0) # qw = 1.0

    control = jnp.array([0.0, 0.0, 0.0]) # zero thrust
    dt = 0.1

    # Drag would be applied but initial velocity is zero, so accel = gravity
    next_state = rocket_step(state, control, dt, params)

    # Expected accel in z is -9.81
    # expected v_z = -9.81 * 0.1 = -0.981
    # expected p_z = 0 + v_z * 0.1 = -0.0981

    pos = next_state[:3]
    vel = next_state[3:6]

    np.testing.assert_allclose(pos, [0.0, 0.0, -0.0981], atol=1e-5)
    np.testing.assert_allclose(vel, [0.0, 0.0, -0.981], atol=1e-5)

def test_rocket_step_hover():
    params = RocketParams(mass=100.0, gravity=9.81, thrust_max=1000.0, thrust_min=0.0, damping=(0.0, 0.0, 0.0))
    # Required hover thrust force is mg
    mg = params.mass * params.gravity
    hover_thrust_frac = mg / params.thrust_max

    state = hover_state(altitude=10.0)[:13]

    control = jnp.array([0.0, 0.0, hover_thrust_frac])
    dt = 0.1

    next_state = rocket_step(state, control, dt, params)

    # State should remain completely unchanged
    np.testing.assert_allclose(next_state, state, atol=1e-5)


from tvc.dynamics import state_to_observation

def test_state_to_observation_shape_and_content():
    # Construct a full 17-element state
    # pos(3), vel(3), quat(4), omega(3), gimbal_angles(2), gimbal_vels(2)
    state = jnp.concatenate([
        jnp.array([1.0, 2.0, 3.0]),          # pos
        jnp.array([0.1, 0.2, 0.3]),          # vel
        jnp.array([1.0, 0.0, 0.0, 0.0]),     # quat (identity)
        jnp.array([0.01, 0.02, 0.03]),       # omega
        jnp.array([0.05, -0.05]),            # gimbal angles
        jnp.array([0.0, 0.0])                # gimbal velocities
    ])

    target_pos = jnp.array([0.0, 0.0, 10.0])
    target_vel = jnp.array([0.0, 0.0, 0.0])
    target_quat = jnp.array([1.0, 0.0, 0.0, 0.0])

    obs = state_to_observation(state, target_pos, target_vel, target_quat)

    # Check shape
    assert obs.shape == (38,), f"Expected observation shape (38,), got {obs.shape}"

    # Check content mapping
    # pos
    np.testing.assert_allclose(obs[:3], [1.0, 2.0, 3.0], atol=1e-5)
    # vel
    np.testing.assert_allclose(obs[3:6], [0.1, 0.2, 0.3], atol=1e-5)
    # rotation matrix (identity quat -> identity matrix flattened)
    np.testing.assert_allclose(obs[6:15], np.eye(3).flatten(), atol=1e-5)
    # omega
    np.testing.assert_allclose(obs[15:18], [0.01, 0.02, 0.03], atol=1e-5)
    # gimbal angles
    np.testing.assert_allclose(obs[18:20], [0.05, -0.05], atol=1e-5)
    # gimbal velocities
    np.testing.assert_allclose(obs[20:22], [0.0, 0.0], atol=1e-5)
    # target pos
    np.testing.assert_allclose(obs[22:25], target_pos, atol=1e-5)
    # target vel
    np.testing.assert_allclose(obs[25:28], target_vel, atol=1e-5)
    # pos error
    np.testing.assert_allclose(obs[28:31], [1.0, 2.0, -7.0], atol=1e-5)
    # vel error
    np.testing.assert_allclose(obs[31:34], [0.1, 0.2, 0.3], atol=1e-5)
    # distance to target
    dist = np.linalg.norm([1.0, 2.0, -7.0])
    np.testing.assert_allclose(obs[34], dist, atol=1e-5)
    # orientation alignment
    np.testing.assert_allclose(obs[35], 1.0, atol=1e-5)
    # vertical alignment
    np.testing.assert_allclose(obs[36], 1.0, atol=1e-5)
    # omega magnitude
    omega_mag = np.linalg.norm([0.01, 0.02, 0.03])
    np.testing.assert_allclose(obs[37], omega_mag, atol=1e-5)


import mujoco

def test_rocket_params_from_model():
    # Construct a minimal viable MJCF XML that contains the necessary elements
    # to test RocketParams.from_model
    xml = """
    <mujoco model="test_rocket">
        <compiler angle="radian"/>
        <option gravity="0 0 -9.81"/>
        <worldbody>
            <body name="vehicle" pos="0 0 0">
                <geom type="cylinder" size="0.1 1" mass="280"/>
                <site name="vehicle_cg" pos="0 0 0"/>
                <site name="thrust_site" pos="0 0 -1.15"/>
            </body>
        </worldbody>
        <actuator>
            <motor name="thrust_control" joint="tvc_x" gear="0 0 8540 0 0 0"/>
        </actuator>
        <sensor>
            <jointpos joint="tvc_x" name="tvc_x_sensor"/>
        </sensor>
        <!-- Need a joint named tvc_x to read the range -->
        <worldbody>
            <body name="gimbal" pos="0 0 -1.15">
                <geom type="sphere" size="0.1" mass="1"/>
                <joint name="tvc_x" type="hinge" axis="1 0 0" range="-0.14 0.14"/>
            </body>
        </worldbody>
    </mujoco>
    """
    model = mujoco.MjModel.from_xml_string(xml)
    params = RocketParams.from_model(model)

    assert params.mass == 280.0
    assert params.thrust_max == 8540.0
    assert params.thrust_min == 8540.0 * 0.4
    assert np.isclose(params.tvc_limit, 0.14)
    assert np.isclose(params.gravity, 9.81)
    # The distance between vehicle_cg (0,0,0) and thrust_site (0,0,-1.15)
    assert np.isclose(params.arm, 1.15)
