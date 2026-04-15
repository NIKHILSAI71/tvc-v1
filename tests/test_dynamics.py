import pytest
import jax.numpy as jnp
import numpy as np

# Add src to path if needed to find src.tvc.dynamics
import sys
import os
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from tvc.dynamics import quaternion_to_rotation_matrix

@pytest.mark.parametrize(
    "quaternion, expected_matrix",
    [
        # Identity (no rotation)
        (
            [1.0, 0.0, 0.0, 0.0],
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        ),
        # 90 degrees around X-axis
        (
            [np.sqrt(2)/2, np.sqrt(2)/2, 0.0, 0.0],
            [
                [1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0],
                [0.0, 1.0, 0.0],
            ]
        ),
        # 90 degrees around Y-axis
        (
            [np.sqrt(2)/2, 0.0, np.sqrt(2)/2, 0.0],
            [
                [0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
                [-1.0, 0.0, 0.0],
            ]
        ),
        # 90 degrees around Z-axis
        (
            [np.sqrt(2)/2, 0.0, 0.0, np.sqrt(2)/2],
            [
                [0.0, -1.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        ),
        # 180 degrees around X-axis
        (
            [0.0, 1.0, 0.0, 0.0],
            [
                [1.0, 0.0, 0.0],
                [0.0, -1.0, 0.0],
                [0.0, 0.0, -1.0],
            ]
        ),
        # Unnormalized quaternion (should normalize and act as identity)
        (
            [2.0, 0.0, 0.0, 0.0],
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        ),
        # Unnormalized quaternion (should normalize and act as 90 deg around X)
        (
            [2.0, 2.0, 0.0, 0.0],
            [
                [1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0],
                [0.0, 1.0, 0.0],
            ]
        ),
    ],
    ids=[
        "identity",
        "90_deg_x",
        "90_deg_y",
        "90_deg_z",
        "180_deg_x",
        "unnormalized_identity",
        "unnormalized_90_deg_x"
    ]
)
def test_quaternion_to_rotation_matrix(quaternion, expected_matrix):
    q = jnp.array(quaternion)
    expected = jnp.array(expected_matrix)

    actual = quaternion_to_rotation_matrix(q)

    # Use atol to handle minor floating point inaccuracies
    np.testing.assert_allclose(actual, expected, atol=1e-6)
