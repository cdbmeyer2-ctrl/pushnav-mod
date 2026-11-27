"""Measured IMU/telescope and camera/telescope rotational calibration."""

from __future__ import annotations

import math
import numpy as np


def _rotation(value, name="rotation"):
    matrix = np.asarray(value, dtype=float)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError(f"{name} must be a finite 3x3 matrix")
    if not np.allclose(matrix.T @ matrix, np.eye(3), atol=1e-3):
        raise ValueError(f"{name} must be orthonormal")
    if not np.isclose(np.linalg.det(matrix), 1.0, atol=1e-3):
        raise ValueError(f"{name} must have determinant +1")
    return matrix


def _unit(vector):
    vector = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm < 1e-8:
        raise ValueError("Rotation axis is degenerate")
    return vector / norm


def fit_rotation(source_vectors, target_vectors, minimum_angle_deg=10.0):
    """Return proper rotation mapping source vectors onto target vectors.

    Uses the Kabsch/Wahba solution. Rows in each input are corresponding
    measured directions. Requires two nonparallel constraints.
    """
    source = np.asarray(source_vectors, dtype=float)
    target = np.asarray(target_vectors, dtype=float)

    if source.ndim != 2 or target.ndim != 2:
        raise ValueError("Vector inputs must be 2D arrays")
    if source.shape != target.shape or source.shape[1] != 3:
        raise ValueError("Inputs must have matching Nx3 shape")
    if source.shape[0] < 2 or not np.isfinite(source).all() \
            or not np.isfinite(target).all():
        raise ValueError("At least two finite vector pairs are required")

    source = np.array([_unit(v) for v in source])
    target = np.array([_unit(v) for v in target])

    # Reject effectively collinear direction sets.
    #
    # Two nonparallel direction families are sufficient to determine a
    # proper 3D rotation. Their centered point cloud is normally planar,
    # so the *smallest* singular value is expected to be zero. Use the
    # second-largest singular value to detect whether the directions span
    # at least two independent axes.
    source_spread = np.linalg.svd(
        source, compute_uv=False
    )
    target_spread = np.linalg.svd(
        target, compute_uv=False
    )
    threshold = math.sin(math.radians(minimum_angle_deg))
    if (
        source_spread.size < 2
        or target_spread.size < 2
        or source_spread[1] < threshold
        or target_spread[1] < threshold
    ):
        raise ValueError("Calibration vectors lack angular diversity")

    covariance = source.T @ target
    u, _, vt = np.linalg.svd(covariance)
    correction = np.eye(3)
    correction[2, 2] = np.linalg.det(vt.T @ u.T)
    result = vt.T @ correction @ u.T
    _rotation(result, "fitted rotation")

    predicted = (result @ source.T).T
    # atan2(cross magnitude, dot) is well-conditioned near zero degrees.
    # arccos(dot) loses precision when dot is extremely close to 1.
    cross_magnitudes = np.linalg.norm(
        np.cross(predicted, target), axis=1
    )
    dot_products = np.clip(
        np.sum(predicted * target, axis=1), -1.0, 1.0
    )
    errors = np.degrees(
        np.arctan2(cross_magnitudes, dot_products)
    )
    return result, float(np.sqrt(np.mean(errors ** 2))), errors.tolist()


def sensor_to_telescope_from_sweep_axes(sensor_axes, telescope_axes):
    """Fit sensor->telescope from signed, known physical rotation axes.

    sensor_axes: gyro rotation axes in raw sensor coordinates.
    telescope_axes: corresponding signed axes in telescope coordinates.
    Telescope frame convention: x aperture-forward, y transverse,
    z upward. Example known axis targets: altitude=(0,1,0),
    azimuth=(0,0,1), provided the sweep directions were signed correctly.
    """
    matrix, rms, errors = fit_rotation(sensor_axes, telescope_axes)
    if rms > 5.0:
        raise ValueError(f"Sensor alignment residual too high: {rms:.2f} deg")
    return matrix, rms, errors


def _quaternion_matrix(q):
    """Convert scalar-first unit quaternion to body->reference matrix."""
    q = np.asarray(q, dtype=float)
    if q.shape != (4,) or not np.isfinite(q).all():
        raise ValueError("Quaternion must contain four finite values")
    norm = float(np.linalg.norm(q))
    if norm < 1e-8:
        raise ValueError("Quaternion norm is zero")
    w, x, y, z = q / norm
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ])


def _rotation_axis_angle(matrix):
    """Return signed axis and angle in radians for a proper rotation."""
    matrix = _rotation(matrix)
    angle = math.acos(float(np.clip(
        (np.trace(matrix) - 1.0) / 2.0, -1.0, 1.0
    )))
    if angle < math.radians(2.0):
        return None, angle

    axis = np.array([
        matrix[2, 1] - matrix[1, 2],
        matrix[0, 2] - matrix[2, 0],
        matrix[1, 0] - matrix[0, 1],
    ]) / (2.0 * math.sin(angle))
    return _unit(axis), angle


def camera_from_telescope_from_pairs(
    imu_quaternions,
    camera_to_sky_matrices,
    minimum_pair_angle_deg=5.0,
    maximum_rms_deg=3.0,
):
    """Estimate camera<-telescope from paired camera/IMU orientations.

    Each IMU quaternion must represent telescope->IMU-reference, and each
    camera matrix must represent camera->celestial, both at the same frame.
    Relative rotations eliminate the unknown global reference rotation.

    Returns (camera_from_telescope, rms_deg, residuals_deg).
    """
    if len(imu_quaternions) != len(camera_to_sky_matrices):
        raise ValueError("IMU and plate-solve sample counts differ")
    if len(imu_quaternions) < 3:
        raise ValueError("At least three paired orientations are required")

    imu = [_quaternion_matrix(q) for q in imu_quaternions]
    camera = [_rotation(m, "camera_to_sky") for m in camera_to_sky_matrices]

    source_axes = []
    target_axes = []
    min_angle = math.radians(minimum_pair_angle_deg)

    # Every pair contributes a relative-rotation axis correspondence.
    for i in range(len(imu)):
        for j in range(i + 1, len(imu)):
            imu_relative = imu[i].T @ imu[j]
            camera_relative = camera[i].T @ camera[j]

            imu_axis, imu_angle = _rotation_axis_angle(imu_relative)
            camera_axis, camera_angle = _rotation_axis_angle(camera_relative)

            if imu_axis is None or camera_axis is None:
                continue
            if min(imu_angle, camera_angle) < min_angle:
                continue

            # Axis mapping: camera_axis = A @ telescope_axis.
            source_axes.append(imu_axis)
            target_axes.append(camera_axis)

    if len(source_axes) < 2:
        raise ValueError(
            "Insufficient rotational motion; capture varied altitude "
            "and azimuth orientations"
        )

    matrix, rms, errors = fit_rotation(source_axes, target_axes)
    if rms > maximum_rms_deg:
        raise ValueError(
            f"Camera alignment residual {rms:.2f} deg exceeds "
            f"{maximum_rms_deg:.2f} deg"
        )
    return matrix, rms, errors



