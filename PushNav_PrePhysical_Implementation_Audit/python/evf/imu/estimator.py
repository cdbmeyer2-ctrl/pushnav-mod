"""Quaternion-based relative attitude estimator for PushNav IMU.

The estimator reports relative telescope-axis motion:
  * altitude_relative_deg is the empirically observed pitch channel
  * azimuth_relative_deg is the empirically observed yaw channel

The first valid accelerometer sample establishes a zero-tilt reference. Yaw is
zeroed at startup because the MPU-6050 has no magnetometer. It will drift.

The sensor_to_telescope matrix is still a provisional proper rotation. The
current tests establish the signs of the pitch/yaw output channels, but do not
fully identify all three breakout-board axes.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

from .protocol import RawSample
from .calibration import Calibration


@dataclass
class Attitude:
    roll_deg: float = 0.0
    pitch_deg: float = 0.0
    yaw_relative_deg: float = 0.0
    altitude_relative_deg: float = 0.0
    azimuth_relative_deg: float = 0.0
    yaw_drift_warning: bool = True
    sample_age_s: float = 0.0
    accel_correction_used: bool = False
    reference_initialized: bool = False


def _dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def _cross(a, b):
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _mat_vec(m, v):
    return tuple(sum(m[r][c] * v[c] for c in range(3)) for r in range(3))


def _det3(m):
    return (
        m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
    )


def _validate_rotation(m):
    if len(m) != 3 or any(len(row) != 3 for row in m):
        raise ValueError("sensor_to_telescope must be a 3x3 matrix")
    rows = [tuple(float(x) for x in row) for row in m]
    for i in range(3):
        for j in range(3):
            target = 1.0 if i == j else 0.0
            if abs(_dot(rows[i], rows[j]) - target) > 1e-3:
                raise ValueError("sensor_to_telescope rows must be orthonormal")
    if abs(_det3(rows) - 1.0) > 1e-3:
        raise ValueError("sensor_to_telescope must have determinant +1 (proper rotation)")
    return tuple(rows)


class RelativeAttitudeEstimator:
    """Integrate bias-corrected gyro data with accelerometer tilt correction.

    The first valid accelerometer sample aligns estimated gravity with the
    local reference +Z. This makes roll/pitch relative to the startup pose;
    it does not claim that startup is physically level. Yaw starts at zero.
    """

    def __init__(
        self,
        calibration: Calibration,
        accel_correction: float | None = None,
        *,
        sensor_to_telescope: Sequence[Sequence[float]] | None = None,
        accel_kp: float = 2.0,
        min_accel_g: float = 0.85,
        max_accel_g: float = 1.15,
        altitude_sign: float = 1.0,
        azimuth_sign: float = 1.0,
    ):
        self.cal = calibration
        # Keep old keyword compatibility; convert the old per-sample factor
        # to a nominal gain at 100 Hz. New code should use accel_kp.
        if accel_correction is not None:
            accel_kp = max(0.0, float(accel_correction) * 100.0)
        self.accel_kp = float(accel_kp)
        self.min_accel_g = float(min_accel_g)
        self.max_accel_g = float(max_accel_g)
        self.altitude_sign = 1.0 if float(altitude_sign) >= 0 else -1.0
        self.azimuth_sign = 1.0 if float(azimuth_sign) >= 0 else -1.0
        self.sensor_to_telescope = _validate_rotation(
            sensor_to_telescope if sensor_to_telescope is not None
            else ((1, 0, 0), (0, 1, 0), (0, 0, 1))
        )
        # q maps telescope-body vectors into the fixed local reference frame.
        self.q = [1.0, 0.0, 0.0, 0.0]  # w, x, y, z
        self._initialized = False
        self._reference_q = None
        self.attitude = Attitude()
        self._last_timestamp: int | None = None
        self._last_timestamp_unwrapped: bool | None = None

    @staticmethod
    def _wrap180(deg: float) -> float:
        return (deg + 180.0) % 360.0 - 180.0

    @staticmethod
    def _normalize(v):
        n = math.sqrt(_dot(v, v))
        if n <= 1e-12:
            return None
        return tuple(x / n for x in v)

    @staticmethod
    def _q_normalize(q):
        n = math.sqrt(sum(x * x for x in q))
        if n <= 1e-12:
            return [1.0, 0.0, 0.0, 0.0]
        return [x / n for x in q]

    @staticmethod
    def _q_slerp(q0, q1, fraction):
        """Interpolate unit quaternions along the shortest rotational path."""
        if not 0.0 <= fraction <= 1.0:
            raise ValueError("fraction must be between 0 and 1")

        a = RelativeAttitudeEstimator._q_normalize(q0)
        b = RelativeAttitudeEstimator._q_normalize(q1)
        dot = sum(x * y for x, y in zip(a, b))

        if dot < 0.0:
            b = [-x for x in b]
            dot = -dot

        dot = max(-1.0, min(1.0, dot))

        if dot > 0.9995:
            result = [x + fraction * (y - x) for x, y in zip(a, b)]
            return RelativeAttitudeEstimator._q_normalize(result)

        angle = math.acos(dot)
        sin_angle = math.sin(angle)
        weight_a = math.sin((1.0 - fraction) * angle) / sin_angle
        weight_b = math.sin(fraction * angle) / sin_angle

        return RelativeAttitudeEstimator._q_normalize([
            weight_a * x + weight_b * y for x, y in zip(a, b)
        ])
    @staticmethod
    def _q_multiply(a, b):
        aw, ax, ay, az = a
        bw, bx, by, bz = b
        return [
            aw*bw - ax*bx - ay*by - az*bz,
            aw*bx + ax*bw + ay*bz - az*by,
            aw*by - ax*bz + ay*bw + az*bx,
            aw*bz + ax*by - ay*bx + az*bw,
        ]

    @staticmethod
    def _q_rotate(q, v):
        p = [0.0, v[0], v[1], v[2]]
        qc = [q[0], -q[1], -q[2], -q[3]]
        out = RelativeAttitudeEstimator._q_multiply(
            RelativeAttitudeEstimator._q_multiply(q, p), qc
        )
        return (out[1], out[2], out[3])

    @classmethod
    def _q_between(cls, source, target):
        """Return q that rotates unit vector source onto unit vector target."""
        a = cls._normalize(source)
        b = cls._normalize(target)
        if a is None or b is None:
            return [1.0, 0.0, 0.0, 0.0]
        d = max(-1.0, min(1.0, _dot(a, b)))
        if d < -0.999999:
            axis = _cross(a, (1.0, 0.0, 0.0))
            if _dot(axis, axis) < 1e-8:
                axis = _cross(a, (0.0, 1.0, 0.0))
            axis = cls._normalize(axis)
            return [0.0, axis[0], axis[1], axis[2]]
        c = _cross(a, b)
        return cls._q_normalize([1.0 + d, c[0], c[1], c[2]])

    def _euler(self):
        # Report rotation relative to the first valid gravity reference.
        q = self.q
        if self._reference_q is not None:
            reference_inverse = [
                self._reference_q[0],
                -self._reference_q[1],
                -self._reference_q[2],
                -self._reference_q[3],
            ]
            q = self._q_normalize(
                self._q_multiply(reference_inverse, self.q)
            )

        w, x, y, z = q
        sinr = 2.0 * (w*x + y*z)
        cosr = 1.0 - 2.0 * (x*x + y*y)
        roll = math.atan2(sinr, cosr)

        sinp = 2.0 * (w*y - z*x)
        sinp = max(-1.0, min(1.0, sinp))
        pitch = math.asin(sinp)

        siny = 2.0 * (w*z + x*y)
        cosy = 1.0 - 2.0 * (y*y + z*z)
        yaw = math.atan2(siny, cosy)
        return tuple(math.degrees(a) for a in (roll, pitch, yaw))

    def update(
        self,
        sample: RawSample,
        device_time_us: int | None = None,
    ) -> Attitude:
        # The stream can supply an unwrapped Arduino timestamp. Direct callers
        # may omit it and retain the original uint32 rollover behavior.
        use_unwrapped = device_time_us is not None
        if use_unwrapped:
            if not isinstance(device_time_us, int) or device_time_us < 0:
                raise ValueError("device_time_us must be a non-negative integer")
            timestamp = device_time_us
        else:
            timestamp = sample.timestamp_us

        if (
            self._last_timestamp is None
            or self._last_timestamp_unwrapped != use_unwrapped
        ):
            dt = 0.0
            self._last_timestamp = timestamp
            self._last_timestamp_unwrapped = use_unwrapped
        else:
            if use_unwrapped:
                delta = timestamp - self._last_timestamp
            else:
                delta = (timestamp - self._last_timestamp) & 0xFFFFFFFF

            if delta <= 0:
                # Ignore duplicate or backward timestamps.
                dt = 0.0
            else:
                dt = delta / 1_000_000.0
                self._last_timestamp = timestamp
                if dt > 0.25:
                    dt = 0.0

        # Bias is measured in raw sensor coordinates; subtract before rotating.
        bx, by, bz = self.cal.gyro_bias_dps
        gx_s, gy_s, gz_s = sample.gyro_dps
        gyro_sensor = (gx_s - bx, gy_s - by, gz_s - bz)
        gyro_body_dps = _mat_vec(self.sensor_to_telescope, gyro_sensor)
        gyro_body = tuple(math.radians(v) for v in gyro_body_dps)

        accel_body = _mat_vec(self.sensor_to_telescope, sample.accel_g)
        norm = math.sqrt(_dot(accel_body, accel_body))
        accel_used = False
        measured_up = None
        if self.min_accel_g <= norm <= self.max_accel_g:
            measured_up = self._normalize(accel_body)

        # Initialize tilt from gravity. Yaw is deliberately arbitrary/zero
        # because an accelerometer cannot determine heading.
        if not self._initialized and measured_up is not None:
            self.q = self._q_between(measured_up, (0.0, 0.0, 1.0))
            self._reference_q = list(self.q)
            self._initialized = True
            dt = 0.0  # do not integrate motion across the reference instant

        if self._initialized and dt > 0.0:
            omega = gyro_body
            if measured_up is not None:
                estimated_up = self._q_rotate(
                    [self.q[0], -self.q[1], -self.q[2], -self.q[3]],
                    (0.0, 0.0, 1.0),
                )
                # Mahony-style proportional gravity correction in this
                # quaternion convention: measured gravity cross estimated.
                error = _cross(measured_up, estimated_up)
                omega = tuple(omega[i] + self.accel_kp * error[i] for i in range(3))
                accel_used = True

            omega_q = [0.0, omega[0], omega[1], omega[2]]
            q_dot = self._q_multiply(self.q, omega_q)
            self.q = self._q_normalize([
                self.q[i] + 0.5 * q_dot[i] * dt for i in range(4)
            ])

        roll, pitch, yaw = self._euler()
        roll = self._wrap180(roll)
        pitch = self._wrap180(pitch)
        yaw = self._wrap180(yaw)
        self.attitude.roll_deg = roll
        self.attitude.pitch_deg = pitch
        self.attitude.yaw_relative_deg = yaw
        self.attitude.altitude_relative_deg = self._wrap180(self.altitude_sign * pitch)
        self.attitude.azimuth_relative_deg = self._wrap180(self.azimuth_sign * yaw)
        self.attitude.yaw_drift_warning = True
        self.attitude.sample_age_s = 0.0
        self.attitude.accel_correction_used = accel_used
        self.attitude.reference_initialized = self._initialized
        return self.attitude

    def reset_reference(self):
        """Zero relative attitude at the next valid stationary gravity sample."""
        self.q = [1.0, 0.0, 0.0, 0.0]
        self._initialized = False
        self._reference_q = None
        self._last_timestamp = None
        self._last_timestamp_unwrapped = None
        self.attitude = Attitude()

    def set_sensor_to_telescope(self, matrix):
        """Change axis mapping; resets the relative reference."""
        self.sensor_to_telescope = _validate_rotation(matrix)
        self.reset_reference()
