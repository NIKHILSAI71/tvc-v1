import jax.numpy as jnp
import numpy as np

from src.tvc.dynamics import quaternion_to_rotation_matrix

def test_quaternion_to_rotation_matrix_identity():
    q = jnp.array([1.0, 0.0, 0.0, 0.0])
    expected = jnp.eye(3)
    actual = quaternion_to_rotation_matrix(q)
    np.testing.assert_allclose(actual, expected, atol=1e-6)

def test_quaternion_to_rotation_matrix_x_90():
    # 90 degree rotation about X axis: q = [cos(pi/4), sin(pi/4), 0, 0]
    angle = np.pi / 2
    q = jnp.array([np.cos(angle/2), np.sin(angle/2), 0.0, 0.0])
    expected = jnp.array([
        [1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
        [0.0, 1.0, 0.0]
    ])
    actual = quaternion_to_rotation_matrix(q)
    np.testing.assert_allclose(actual, expected, atol=1e-6)

def test_quaternion_to_rotation_matrix_y_90():
    # 90 degree rotation about Y axis: q = [cos(pi/4), 0, sin(pi/4), 0]
    angle = np.pi / 2
    q = jnp.array([np.cos(angle/2), 0.0, np.sin(angle/2), 0.0])
    expected = jnp.array([
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0]
    ])
    actual = quaternion_to_rotation_matrix(q)
    np.testing.assert_allclose(actual, expected, atol=1e-6)

def test_quaternion_to_rotation_matrix_z_90():
    # 90 degree rotation about Z axis: q = [cos(pi/4), 0, 0, sin(pi/4)]
    angle = np.pi / 2
    q = jnp.array([np.cos(angle/2), 0.0, 0.0, np.sin(angle/2)])
    expected = jnp.array([
        [0.0, -1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0]
    ])
    actual = quaternion_to_rotation_matrix(q)
    np.testing.assert_allclose(actual, expected, atol=1e-6)

def test_quaternion_to_rotation_matrix_unnormalized():
    # Unnormalized identity
    q = jnp.array([2.0, 0.0, 0.0, 0.0])
    expected = jnp.eye(3)
    actual = quaternion_to_rotation_matrix(q)
    np.testing.assert_allclose(actual, expected, atol=1e-6)

def test_quaternion_to_rotation_matrix_arbitrary():
    # Arbitrary unnormalized quaternion
    q_unnorm = jnp.array([1.0, 2.0, 3.0, 4.0])
    norm = jnp.linalg.norm(q_unnorm)
    q = q_unnorm / norm

    # Pre-calculated expected values from scipy.spatial.transform.Rotation
    # r = Rotation.from_quat([x, y, z, w])  # Note scipy uses scalar-last (x,y,z,w)
    # expected = r.as_matrix()
    expected = jnp.array([
        [-0.66666667,  0.13333333,  0.73333333],
        [ 0.66666667, -0.33333333,  0.66666667],
        [ 0.33333333,  0.93333333,  0.13333333]
    ])

    # Function normalizes internally, so we can pass unnormalized or normalized
    actual_unnorm = quaternion_to_rotation_matrix(q_unnorm)
    actual_norm = quaternion_to_rotation_matrix(q)

    np.testing.assert_allclose(actual_unnorm, expected, atol=1e-6)
    np.testing.assert_allclose(actual_norm, expected, atol=1e-6)

def test_quaternion_to_rotation_matrix_negative_w():
    # Negative w, should represent the same rotation as positive w
    q1 = jnp.array([0.5, 0.5, 0.5, 0.5])
    q2 = jnp.array([-0.5, -0.5, -0.5, -0.5])

    actual1 = quaternion_to_rotation_matrix(q1)
    actual2 = quaternion_to_rotation_matrix(q2)

    np.testing.assert_allclose(actual1, actual2, atol=1e-6)
