"""Synthetic regression tests for IMU/camera/telescope rotational calibration."""

import math
from types import SimpleNamespace

import numpy as np
import pytest

from evf.imu.alignment_calibration import (
    camera_from_telescope_from_pairs,
    fit_rotation,
    sensor_to_telescope_from_sweep_axes,
)
from evf.engine.engine import Engine


def _axis_angle_matrix(axis, angle_deg):
    axis = np.asarray(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    angle = math.radians(angle_deg)
    x, y, z = axis
    c = math.cos(angle)
    s = math.sin(angle)
    t = 1.0 - c
    return np.array([
        [t*x*x + c,   t*x*y - s*z, t*x*z + s*y],
        [t*x*y + s*z, t*y*y + c,   t*y*z - s*x],
        [t*x*z - s*y, t*y*z + s*x, t*z*z + c],
    ])


def _matrix_to_quaternion(matrix):
    """Convert a proper rotation matrix to scalar-first wxyz quaternion."""
    m = np.asarray(matrix, dtype=float)
    trace = float(np.trace(m))

    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        q = (
            0.25 * s,
            (m[2, 1] - m[1, 2]) / s,
            (m[0, 2] - m[2, 0]) / s,
            (m[1, 0] - m[0, 1]) / s,
        )
    else:
        i = int(np.argmax(np.diag(m)))
        if i == 0:
            s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
            q = (
                (m[2, 1] - m[1, 2]) / s,
                0.25 * s,
                (m[0, 1] + m[1, 0]) / s,
                (m[0, 2] + m[2, 0]) / s,
            )
        elif i == 1:
            s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
            q = (
                (m[0, 2] - m[2, 0]) / s,
                (m[0, 1] + m[1, 0]) / s,
                0.25 * s,
                (m[1, 2] + m[2, 1]) / s,
            )
        else:
            s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
            q = (
                (m[1, 0] - m[0, 1]) / s,
                (m[0, 2] + m[2, 0]) / s,
                (m[1, 2] + m[2, 1]) / s,
                0.25 * s,
            )

    q = np.asarray(q, dtype=float)
    return tuple(q / np.linalg.norm(q))


def _make_synthetic_camera_pairs(camera_from_telescope):
    """Build matched telescope->reference IMU and camera->sky rotations."""
    telescope_orientations = [
        np.eye(3),
        _axis_angle_matrix((1, 0, 0), 23.0),
        _axis_angle_matrix((0, 1, 0), 31.0),
        _axis_angle_matrix((0, 0, 1), 19.0),
        _axis_angle_matrix((1, 1, 0), 42.0),
        _axis_angle_matrix((0, 1, 1), -37.0),
        _axis_angle_matrix((1, 2, 3), 54.0),
    ]

    # Camera->sky = telescope->sky @ inverse(camera<-telescope).
    # Thus C_i.T @ C_j = A @ (T_i.T @ T_j) @ A.T,
    # and relative-rotation axes map by camera_from_telescope A.
    camera_to_sky = [
        t @ camera_from_telescope.T
        for t in telescope_orientations
    ]
    imu_quaternions = [
        _matrix_to_quaternion(t)
        for t in telescope_orientations
    ]
    return imu_quaternions, camera_to_sky


def test_fit_rotation_recovers_known_proper_rotation_from_planar_vectors():
    expected = _axis_angle_matrix((1.0, 2.0, -1.0), 67.0)
    source = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
    ])
    target = (expected @ source.T).T

    fitted, rms, residuals = fit_rotation(source, target)

    np.testing.assert_allclose(fitted, expected, atol=1e-7)
    assert rms < 1e-6
    assert max(residuals) < 1e-6
    assert np.linalg.det(fitted) == pytest.approx(1.0, abs=1e-7)


def test_fit_rotation_rejects_single_axis_only():
    source = np.array([
        [1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
    ])
    target = source.copy()

    with pytest.raises(ValueError, match="angular diversity"):
        fit_rotation(source, target)


def test_sensor_sweep_fit_recovers_known_sensor_to_telescope_rotation():
    expected = _axis_angle_matrix((0.3, -0.8, 0.5), 42.0)
    sensor_axes = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    telescope_axes = (expected @ sensor_axes.T).T

    fitted, rms, errors = sensor_to_telescope_from_sweep_axes(
        sensor_axes, telescope_axes
    )

    np.testing.assert_allclose(fitted, expected, atol=1e-7)
    assert rms < 1e-6
    assert max(errors) < 1e-6


def test_camera_from_telescope_fit_recovers_known_mounting_rotation():
    expected = _axis_angle_matrix((0.4, 0.2, 0.9), 38.0)
    imu_q, camera_to_sky = _make_synthetic_camera_pairs(expected)

    fitted, rms, residuals = camera_from_telescope_from_pairs(
        imu_q,
        camera_to_sky,
        maximum_rms_deg=0.1,
    )

    np.testing.assert_allclose(fitted, expected, atol=1e-6)
    assert rms < 1e-5
    assert max(residuals) < 1e-5


def test_synthetic_plate_solve_propagation_uses_aperture_plus_x():
    """A synthetic solved anchor moves the +X optical axis by known RA."""
    anchor = SimpleNamespace(
        quaternion=(1.0, 0.0, 0.0, 0.0),
        telescope_to_sky=np.eye(3),
    )

    angle = math.radians(12.0)
    # Body +X rotates toward +Y around +Z; identity telescope->sky
    # therefore gives RA=12 degrees, Dec=0.
    current_q = (
        math.cos(angle / 2.0),
        0.0,
        0.0,
        math.sin(angle / 2.0),
    )

    ra_dec = Engine._imu_propagate_radec(anchor, current_q)

    assert ra_dec is not None
    assert ra_dec[0] == pytest.approx(12.0, abs=1e-7)
    assert ra_dec[1] == pytest.approx(0.0, abs=1e-7)
