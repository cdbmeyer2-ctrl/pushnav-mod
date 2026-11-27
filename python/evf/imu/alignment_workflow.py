"""Evidence-based calibration workflow for the PushNav telescope IMU."""

from __future__ import annotations

from dataclasses import dataclass
import math
import json
from pathlib import Path
import threading
import time

import numpy as np

from evf.imu.alignment_calibration import (
    camera_from_telescope_from_pairs,
    sensor_to_telescope_from_sweep_axes,
)
from evf.solver.sync import orientation_from_radec_roll


@dataclass(frozen=True)
class SensorSweep:
    axis: str
    direction: int
    sensor_axis: tuple[float, float, float]
    sample_count: int
    duration_s: float
    integrated_angle_deg: float
    gyro_noise_dps: float
    started_at: float
    finished_at: float


@dataclass(frozen=True)
class CameraPair:
    frame_id: int
    frame_timestamp: float
    device_time_us: int
    quaternion: tuple[float, float, float, float]
    camera_to_sky: tuple[tuple[float, float, float], ...]
    timestamp_quality: str = "untrusted"
    clock_sync_verified: bool = False
    timing_uncertainty_s: float | None = None
    timestamp_source: str = "untrusted"
    exposure_midpoint_verified: bool = False
    evidence_origin: str = "untrusted"


class ImuAlignmentWorkflow:
    """Collect and validate actual physical alignment measurements."""

    MIN_SWEEP_SAMPLES = 50
    MIN_SWEEP_ANGLE_DEG = 8.0
    MIN_SWEEP_DURATION_S = 1.0
    MAX_SWEEP_DURATION_S = 30.0
    MAX_SWEEP_NOISE_DPS = 3.0
    MIN_CAMERA_PAIRS = 4
    MAX_SENSOR_RMS_DEG = 5.0
    MAX_CAMERA_RMS_DEG = 3.0
    MIN_CAMERA_PAIR_SEPARATION_DEG = 5.0

    def __init__(self, service):
        self.service = service
        self._lock = threading.RLock()
        self._active_sweep = None
        self._sweeps = []
        self._camera_pairs = []
        self._sensor_fit = None
        self._camera_fit = None

    @staticmethod
    def _timing_is_verified(item) -> bool:
        """Only accept independently verified exposure-midpoint timestamps.

        A DirectShow SampleCB/QPC callback timestamp establishes a host clock
        domain, not the time photons were integrated. It is never sufficient
        evidence for celestial propagation.
        """
        uncertainty = getattr(item, "timing_uncertainty_s", None)
        try:
            uncertainty = float(uncertainty)
        except (TypeError, ValueError):
            return False
        return bool(
            getattr(item, "timestamp_quality", "untrusted") == "exposure_midpoint"
            and getattr(item, "clock_sync_verified", False) is True
            and getattr(item, "exposure_midpoint_verified", False) is True
            and getattr(item, "evidence_origin", "untrusted") == "physical"
            and math.isfinite(uncertainty)
            and 0.0 <= uncertainty <= 0.010
        )

    def status(self):
        with self._lock:
            return {
                "active_sweep": (
                    {
                        "axis": self._active_sweep["axis"],
                        "direction": self._active_sweep["direction"],
                        "elapsed_s": max(
                            0.0, time.monotonic() -
                            self._active_sweep["started_at"]
                        ),
                    }
                    if self._active_sweep else None
                ),
                "sensor_sweep_count": len(self._sweeps),
                "sensor_sweeps": [
                    {
                        "axis": item.axis,
                        "direction": item.direction,
                        "sample_count": item.sample_count,
                        "duration_s": item.duration_s,
                        "integrated_angle_deg": item.integrated_angle_deg,
                        "gyro_noise_dps": item.gyro_noise_dps,
                    }
                    for item in self._sweeps
                ],
                "camera_pair_count": len(self._camera_pairs),
                "trusted_camera_pair_count": sum(
                    self._timing_is_verified(item)
                    for item in self._camera_pairs
                ),
                "sensor_fit": self._sensor_fit,
                "camera_fit": self._camera_fit,
                "camera_timing_ready": (
                    len(self._camera_pairs) >= self.MIN_CAMERA_PAIRS
                    and all(
                        item.timestamp_quality == "callback"
                        and item.clock_sync_verified
                        for item in self._camera_pairs
                    )
                ),
                "ready_to_finalize": (
                    self._sensor_fit is not None
                    and self._camera_fit is not None
                    and len(self._camera_pairs) >= self.MIN_CAMERA_PAIRS
                    and all(
                        item.timestamp_quality == "callback"
                        and item.clock_sync_verified
                        for item in self._camera_pairs
                    )
                ),
            }

    def start_sensor_sweep(self, axis: str, direction: int):
        axis = str(axis).lower()
        if axis not in ("altitude", "azimuth"):
            raise ValueError("axis must be altitude or azimuth")
        if direction not in (-1, 1):
            raise ValueError("direction must be -1 or +1")

        if self.service is None or self.service.stream is None:
            raise RuntimeError("IMU stream is not running")

        with self._lock:
            if self._active_sweep is not None:
                raise RuntimeError("A sensor sweep is already active")

            self._active_sweep = {
                "axis": axis,
                "direction": int(direction),
                "started_at": time.monotonic(),
                "device_start_us": (
                    self.service.stream.history.latest().device_time_us
                    if self.service.stream.history.latest() else None
                ),
            }

    def finish_sensor_sweep(self):
        with self._lock:
            active = self._active_sweep
            if active is None:
                raise RuntimeError("No sensor sweep is active")
            self._active_sweep = None

        finished = time.monotonic()
        duration = finished - active["started_at"]
        if not self.MIN_SWEEP_DURATION_S <= duration <= self.MAX_SWEEP_DURATION_S:
            raise ValueError(
                f"Sweep duration must be between "
                f"{self.MIN_SWEEP_DURATION_S:.0f} and "
                f"{self.MAX_SWEEP_DURATION_S:.0f} seconds"
            )

        records = self.service.stream.history.samples_between(
            active["started_at"], finished
        )
        if len(records) < self.MIN_SWEEP_SAMPLES:
            raise ValueError(
                f"Insufficient sweep samples: {len(records)}; "
                f"need at least {self.MIN_SWEEP_SAMPLES}"
            )

        # Use the device clock for integration; receipt timestamps select
        # the capture window but are not substituted for device time.
        records = sorted(records, key=lambda item: item.device_time_us)
        times = np.asarray(
            [item.device_time_us for item in records], dtype=float
        ) / 1_000_000.0
        gyro = np.asarray(
            [item.sample.gyro_dps for item in records], dtype=float
        )

        if not np.isfinite(gyro).all():
            raise ValueError("Sweep contains non-finite gyro samples")

        dt = np.diff(times)
        if np.any(dt <= 0) or np.any(dt > 0.1):
            raise ValueError("Sweep has invalid or discontinuous device timestamps")

        # Use the independently measured stationary gyro calibration. Do not
        # estimate bias from the sweep itself: a telescope still moving at
        # either endpoint would otherwise be mistaken for sensor bias.
        calibration_path = Path(self.service.config_dir) / "imu_calibration.json"
        try:
            with calibration_path.open("r", encoding="utf-8-sig") as handle:
                calibration = json.load(handle)
            bias = np.asarray(calibration["gyro_bias_dps"], dtype=float)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise RuntimeError(
                "Run stationary gyroscope calibration before axis sweeps"
            ) from exc
        if bias.shape != (3,) or not np.isfinite(bias).all():
            raise ValueError("Saved gyro bias must contain three finite values")
        corrected = gyro - bias

        edge_count = max(10, len(gyro) // 10)
        edge_magnitudes = np.linalg.norm(
            np.vstack((corrected[:edge_count], corrected[-edge_count:])),
            axis=1,
        )
        if (
            float(np.median(edge_magnitudes)) > 2.0
            or float(np.percentile(edge_magnitudes, 90)) > 5.0
        ):
            raise ValueError(
                "Keep the telescope stationary at the beginning and end "
                "of each sweep"
            )

        magnitudes = np.linalg.norm(corrected, axis=1)
        moving = magnitudes >= 1.0
        if int(np.count_nonzero(moving)) < self.MIN_SWEEP_SAMPLES // 2:
            raise ValueError(
                "Not enough moving gyro samples; make a deliberate, "
                "continuous telescope sweep"
            )

        moving_vectors = corrected[moving]
        # Robustly estimate the signed axis from all moving samples. Reject
        # sweeps that reverse or wander rather than averaging them into a
        # misleading direction.
        axis_vector = np.median(moving_vectors, axis=0)
        axis_norm = float(np.linalg.norm(axis_vector))
        if axis_norm < 1.0:
            raise ValueError("Measured sweep axis is too weak")

        sensor_axis = (axis_vector * active["direction"]) / axis_norm
        projections = corrected @ sensor_axis
        moving_projection = projections[moving]
        aligned_fraction = float(np.mean(
            moving_projection * active["direction"] > 0.0
        ))
        if aligned_fraction < 0.85:
            raise ValueError(
                "Sweep direction reversed or was inconsistent; capture one "
                "continuous motion in the requested direction"
            )
        perpendicular = corrected - projections[:, None] * sensor_axis
        gyro_noise = float(np.median(np.linalg.norm(
            perpendicular[moving], axis=1
        )))

        if gyro_noise > self.MAX_SWEEP_NOISE_DPS:
            raise ValueError(
                f"Sweep gyro noise too high ({gyro_noise:.2f} dps)"
            )

        # Integrate angular speed over device-clock intervals. The measured
        # direction is retained in sensor_axis; the target direction is
        # selected explicitly by the operator.
        # Integrate signed angular velocity projected onto the fitted axis.
        # Keep rate and dt arrays aligned across every timestamp interval.
        signed_rate = projections * active["direction"]
        interval_rate = 0.5 * (signed_rate[:-1] + signed_rate[1:])
        integrated = float(np.sum(np.maximum(interval_rate, 0.0) * dt))
        if integrated < self.MIN_SWEEP_ANGLE_DEG:
            raise ValueError(
                f"Sweep moved only {integrated:.1f} degrees; "
                f"need at least {self.MIN_SWEEP_ANGLE_DEG:.0f}"
            )

        result = SensorSweep(
            axis=active["axis"],
            direction=active["direction"],
            sensor_axis=tuple(float(v) for v in sensor_axis),
            sample_count=len(records),
            duration_s=duration,
            integrated_angle_deg=integrated,
            gyro_noise_dps=gyro_noise,
            started_at=active["started_at"],
            finished_at=finished,
        )

        with self._lock:
            self._sweeps.append(result)
            self._sensor_fit = None

        return self.status()

    def fit_sensor_alignment(self):
        with self._lock:
            sweeps = tuple(self._sweeps)

        for axis in ("altitude", "azimuth"):
            directions = {
                item.direction for item in sweeps if item.axis == axis
            }
            if directions != {-1, 1}:
                raise ValueError(
                    f"Capture both positive and negative {axis} sweeps"
                )

        # Positive and negative telescope movements must identify the
        # same canonical sensor axis. A reversed operator movement would
        # otherwise be silently absorbed into the fitted sensor transform.
        for axis in ("altitude", "azimuth"):
            positive = [
                np.asarray(item.sensor_axis, dtype=float)
                for item in sweeps
                if item.axis == axis and item.direction == 1
            ]
            negative = [
                np.asarray(item.sensor_axis, dtype=float)
                for item in sweeps
                if item.axis == axis and item.direction == -1
            ]

            positive_axis = np.mean(positive, axis=0)
            negative_axis = np.mean(negative, axis=0)

            positive_norm = float(np.linalg.norm(positive_axis))
            negative_norm = float(np.linalg.norm(negative_axis))

            if positive_norm < 1e-8 or negative_norm < 1e-8:
                raise ValueError(
                    f"Insufficient directional consistency for {axis}"
                )

            alignment = float(
                np.dot(
                    positive_axis / positive_norm,
                    negative_axis / negative_norm,
                )
            )

            if alignment < 0.90:
                raise ValueError(
                    f"Positive and negative {axis} sweeps disagree "
                    f"(axis alignment {alignment:.3f}); repeat the "
                    "sweeps in their requested directions"
                )

        source = []
        target = []
        for item in sweeps:
            source.append(item.sensor_axis)
            if item.axis == "altitude":
                # Right-handed telescope frame: raising the aperture is -Y.
                target.append((0.0, -1.0, 0.0))
            else:
                # Positive azimuth is CCW viewed from above (+Z).
                target.append((0.0, 0.0, 1.0))

        matrix, rms, errors = sensor_to_telescope_from_sweep_axes(
            source, target
        )
        if rms > self.MAX_SENSOR_RMS_DEG:
            raise ValueError("Sensor alignment residual exceeds limit")

        result = {
            "matrix": matrix.tolist(),
            "rms_deg": float(rms),
            "residuals_deg": [float(v) for v in errors],
            "sweep_count": len(sweeps),
            "sample_count": sum(item.sample_count for item in sweeps),
            "axes": ["altitude", "azimuth"],
        }
        with self._lock:
            self._sensor_fit = result
        return result

    def capture_camera_pair(
        self, pointing, frame_timestamp: float, frame_id: int, frame_metadata=None
    ):
        """Pair a solve with IMU orientation at a synchronized camera callback time."""
        if self.service is None:
            raise RuntimeError("IMU service is not running")
        if frame_metadata is None:
            raise ValueError("Camera frame has no timestamp provenance")
        if not self._timing_is_verified(frame_metadata):
            raise ValueError(
                "Camera exposure-midpoint timing is not physically verified "
                "to <=10 ms; a DirectShow callback/QPC timestamp is insufficient"
            )

        orientation = self.service.orientation_at_host_time(frame_timestamp)
        if orientation is None:
            raise ValueError(
                "No initialized IMU orientation brackets this camera frame"
            )

        device_time_us, quaternion = orientation
        camera_matrix = orientation_from_radec_roll(
            pointing.ra_j2000,
            pointing.dec_j2000,
            pointing.roll,
        )

        pair = CameraPair(
            frame_id=int(frame_id),
            frame_timestamp=float(frame_timestamp),
            device_time_us=int(device_time_us),
            quaternion=tuple(float(v) for v in quaternion),
            camera_to_sky=tuple(
                tuple(float(v) for v in row) for row in camera_matrix
            ),
            timestamp_quality=frame_metadata.timestamp_quality,
            clock_sync_verified=bool(frame_metadata.clock_sync_verified),
            timing_uncertainty_s=frame_metadata.timing_uncertainty_s,
            timestamp_source=str(getattr(frame_metadata, "timestamp_source", "untrusted")),
            exposure_midpoint_verified=bool(frame_metadata.exposure_midpoint_verified),
            evidence_origin=str(frame_metadata.evidence_origin),
        )

        with self._lock:
            if any(item.frame_id == pair.frame_id for item in self._camera_pairs):
                return self.status()
            # One pair per materially distinct pointing pose. This prevents a
            # stationary scope producing dozens of nearly identical solves
            # and dominating the camera/telescope rotation fit.
            new_matrix = np.asarray(pair.camera_to_sky, dtype=float)
            for existing in self._camera_pairs:
                old_matrix = np.asarray(existing.camera_to_sky, dtype=float)
                relative = old_matrix.T @ new_matrix
                cosine = float(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))
                separation = math.degrees(math.acos(cosine))
                if separation < self.MIN_CAMERA_PAIR_SEPARATION_DEG:
                    return self.status()
            self._camera_pairs.append(pair)
            self._camera_fit = None

        return self.status()

    def fit_camera_alignment(self):
        with self._lock:
            pairs = tuple(self._camera_pairs)

        if len(pairs) < self.MIN_CAMERA_PAIRS:
            raise ValueError(
                f"Need at least {self.MIN_CAMERA_PAIRS} distinct "
                "plate-solve/IMU pairs"
            )

        # Require genuinely different camera attitudes. Repeated solves
        # without telescope movement are not calibration evidence.
        matrices = [
            np.asarray(item.camera_to_sky, dtype=float) for item in pairs
        ]
        distinct_motion = []
        for i in range(len(matrices)):
            for j in range(i + 1, len(matrices)):
                relative = matrices[i].T @ matrices[j]
                angle = math.degrees(math.acos(float(np.clip(
                    (np.trace(relative) - 1.0) / 2.0, -1.0, 1.0
                ))))
                if angle >= 5.0:
                    distinct_motion.append(angle)

        if len(distinct_motion) < 3:
            raise ValueError(
                "Insufficient camera attitude coverage; move the telescope "
                "to several substantially different altitude/azimuth positions"
            )

        matrix, rms, errors = camera_from_telescope_from_pairs(
            [item.quaternion for item in pairs],
            [item.camera_to_sky for item in pairs],
            maximum_rms_deg=self.MAX_CAMERA_RMS_DEG,
        )

        result = {
            "matrix": matrix.tolist(),
            "rms_deg": float(rms),
            "residuals_deg": [float(v) for v in errors],
            "pair_count": len(pairs),
            "frame_ids": [item.frame_id for item in pairs],
            "distinct_motion_count": len(distinct_motion),
        }
        with self._lock:
            self._camera_fit = result
        return result

    def measured_result(self):
        with self._lock:
            if self._sensor_fit is None or self._camera_fit is None:
                raise ValueError(
                    "Both sensor and camera alignment fits must pass "
                    "before finalization"
                )
            if not self._camera_pairs or any(
                not self._timing_is_verified(item) for item in self._camera_pairs
            ):
                raise ValueError(
                    "Camera exposure-midpoint timing is not physically verified "
                    "to <=10 ms for every pair; callback timestamps cannot be used"
                )

            return {
                "sensor_to_telescope": self._sensor_fit["matrix"],
                "camera_from_telescope": self._camera_fit["matrix"],
                "sensor_evidence": self._sensor_fit,
                "camera_evidence": self._camera_fit,
                "camera_timing_evidence": {
                    "pair_count": len(self._camera_pairs),
                    "frame_ids": [item.frame_id for item in self._camera_pairs],
                    "timestamp_source": self._camera_pairs[0].timestamp_source,
                    "clock_sync_verified": True,
                    "timing_uncertainty_s": max(
                        float(item.timing_uncertainty_s) for item in self._camera_pairs
                    ),
                    "exposure_midpoint_verified": True,
                    "evidence_origin": "physical",
                    "timestamp_quality": "exposure_midpoint",
                },
                "calibration_method": "physical_sweeps_and_paired_plate_solves_v2",
                "created_at_unix": time.time(),
            }

    def reset(self):
        with self._lock:
            if self._active_sweep is not None:
                raise RuntimeError("Finish or cancel the active sweep first")
            self._sweeps.clear()
            self._camera_pairs.clear()
            self._sensor_fit = None
            self._camera_fit = None



