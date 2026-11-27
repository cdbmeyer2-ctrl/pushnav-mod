"""Lifecycle wrapper for the optional telescope IMU."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
import json
import logging
import os
from pathlib import Path
import tempfile
import threading
import time

from evf.imu.calibration import Calibration, calibrate_stationary
from evf.imu.estimator import Attitude, RelativeAttitudeEstimator
from evf.imu.stream import ImuSerialStream
from evf.imu.protocol import RawSample
from evf.imu.interpolation import interpolate_quaternion
from evf.imu.clock_sync import map_host_time_to_device_us

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ImuSnapshot:
    """One synchronized IMU attitude, rotation, and sample timestamp."""

    attitude: Attitude
    quaternion: tuple[float, float, float, float]
    device_time_us: int
    received_at_monotonic: float
    raw_sample: RawSample
    continuity_id: int = 0


class ImuService:
    """Loads IMU configuration and owns the serial reader and estimator."""

    def __init__(self, config_dir: Path, port: str = "COM3"):
        self.config_dir = Path(config_dir)
        self.port = port
        self.stream: ImuSerialStream | None = None
        self.estimator: RelativeAttitudeEstimator | None = None
        # True only when sensor_to_telescope came from a verified alignment.
        # A provisional identity mapping supports relative motion, never sky pointing.
        self.sensor_mapping_verified = False

        self._worker_thread: threading.Thread | None = None
        self._worker_stop_event = threading.Event()
        self._attitude_lock = threading.Lock()
        # Serializes estimator.update()/quaternion reads against calibration.
        self._estimator_lock = threading.Lock()
        self._latest_attitude: Attitude | None = None
        self._latest_snapshot: ImuSnapshot | None = None
        self._orientation_history: deque[ImuSnapshot] = deque(maxlen=2000)
        self._continuity_id = 0
        self._max_contiguous_gap_us = 250_000

    @property
    def latest_attitude(self) -> Attitude | None:
        """Return a safe copy of the most recently estimated attitude."""
        with self._attitude_lock:
            if self._latest_attitude is None:
                return None
            return replace(self._latest_attitude)

    @property
    def latest_snapshot(self) -> ImuSnapshot | None:
        """Return a safe copy of the latest synchronized IMU snapshot."""
        with self._attitude_lock:
            snapshot = self._latest_snapshot
            if snapshot is None:
                return None
            return replace(
                snapshot,
                attitude=replace(snapshot.attitude),
            )

    @property
    def orientation_history(self) -> tuple[ImuSnapshot, ...]:
        """Return a safe chronological copy of recent IMU orientations."""
        with self._attitude_lock:
            return tuple(
                replace(item, attitude=replace(item.attitude))
                for item in self._orientation_history
            )

    def interpolated_quaternion(
        self, device_time_us: int, max_gap_us: int = 30000
    ) -> tuple[float, float, float, float] | None:
        """Return an interpolated orientation at an Arduino device timestamp."""
        history = self.orientation_history
        samples = (
            (item.device_time_us, item.quaternion)
            for item in history
        )
        return interpolate_quaternion(samples, device_time_us, max_gap_us)

    def orientation_at_host_time(
        self, host_monotonic: float, max_gap_us: int = 30000
    ) -> tuple[int, tuple[float, float, float, float]] | None:
        """Estimate orientation at a host time using approximate clock mapping."""
        stream = self.stream
        if stream is None:
            return None

        records = stream.history.snapshot()
        device_time_us = map_host_time_to_device_us(records, host_monotonic)
        if device_time_us is None:
            return None

        history = self.orientation_history
        previous = None
        reference_initialized = False

        for item in history:
            if item.device_time_us == device_time_us:
                reference_initialized = item.attitude.reference_initialized
                break

            if (
                previous is not None
                and previous.device_time_us < device_time_us < item.device_time_us
            ):
                reference_initialized = (
                    previous.attitude.reference_initialized
                    and item.attitude.reference_initialized
                )
                break

            previous = item

        if not reference_initialized:
            return None

        samples = (
            (item.device_time_us, item.quaternion)
            for item in history
        )
        quaternion = interpolate_quaternion(
            samples, device_time_us, max_gap_us
        )
        if quaternion is None:
            return None

        return device_time_us, quaternion

    def set_sensor_to_telescope(self, matrix) -> None:
        """Apply a physically verified sensor mapping without racing the worker."""
        with self._estimator_lock:
            if self.estimator is None:
                raise RuntimeError("IMU estimator is not initialized")
            self.estimator.set_sensor_to_telescope(matrix)
            self.sensor_mapping_verified = True
            with self._attitude_lock:
                self._latest_attitude = None
                self._latest_snapshot = None
                self._orientation_history.clear()

    def start(self) -> bool:
        calibration_path = self.config_dir / "imu_calibration.json"
        alignment_path = self.config_dir / "sensor_alignment.json"

        try:
            with calibration_path.open("r", encoding="utf-8-sig") as f:
                data = json.load(f)

            calibration = Calibration(
                gyro_bias_dps=tuple(data["gyro_bias_dps"]),
                accel_mean_g=tuple(data["accel_mean_g"]),
                gyro_noise_dps=tuple(data["gyro_noise_dps"]),
                sample_count=int(data["sample_count"]),
            )

            # The sensor mounting transform is a physical calibration, not
            # something that can safely be inferred from software.  If it is
            # unavailable, use identity solely to keep relative-motion
            # telemetry alive and explicitly prohibit celestial propagation.
            alignment = {}
            try:
                with alignment_path.open("r", encoding="utf-8-sig") as f:
                    loaded_alignment = json.load(f)
                if isinstance(loaded_alignment, dict):
                    alignment = loaded_alignment
                else:
                    logger.warning(
                        "IMU alignment file is not a JSON object; "
                        "using provisional identity mapping."
                    )
            except FileNotFoundError:
                logger.warning(
                    "IMU alignment file %s is missing; using provisional "
                    "identity mapping. Relative motion only.",
                    alignment_path,
                )
            except (OSError, json.JSONDecodeError) as exc:
                logger.warning(
                    "Could not read IMU alignment (%s); using provisional "
                    "identity mapping. Relative motion only.",
                    exc,
                )

            sensor_matrix = alignment.get("sensor_to_telescope")
            mapping_verified = (
                alignment.get("mapping_status") == "VERIFIED"
                and sensor_matrix is not None
            )

            estimator_options = {
                "accel_kp": alignment.get("accel_kp_per_second", 2.0),
                "min_accel_g": alignment.get("min_accel_g", 0.85),
                "max_accel_g": alignment.get("max_accel_g", 1.15),
                "altitude_sign": alignment.get("altitude_sign", 1.0),
                "azimuth_sign": alignment.get("azimuth_sign", 1.0),
            }

            try:
                self.estimator = RelativeAttitudeEstimator(
                    calibration,
                    sensor_to_telescope=(
                        sensor_matrix
                        if sensor_matrix is not None
                        else ((1, 0, 0), (0, 1, 0), (0, 0, 1))
                    ),
                    **estimator_options,
                )
            except (TypeError, ValueError) as exc:
                # A malformed/unusable mounting matrix must not be silently
                # treated as calibrated. Retry with an explicit provisional
                # identity transform for relative telemetry only.
                logger.warning(
                    "Invalid IMU sensor-to-telescope transform (%s); "
                    "using provisional identity mapping. Relative motion only.",
                    exc,
                )
                self.estimator = RelativeAttitudeEstimator(
                    calibration,
                    sensor_to_telescope=((1, 0, 0), (0, 1, 0), (0, 0, 1)),
                    **estimator_options,
                )
                mapping_verified = False

            self.sensor_mapping_verified = bool(mapping_verified)
            if not self.sensor_mapping_verified:
                logger.warning(
                    "IMU sensor-to-telescope mapping is NOT VERIFIED; "
                    "relative motion only, celestial propagation disabled."
                )
            else:
                logger.info("IMU sensor-to-telescope mapping is VERIFIED.")

            self.stream = ImuSerialStream(
                port=self.port,
                baudrate=115200,
            )
            self.stream.start()

            self._worker_stop_event.clear()
            self._continuity_id += 1
            with self._attitude_lock:
                self._latest_attitude = None
                self._latest_snapshot = None
                self._orientation_history.clear()
            self._worker_thread = threading.Thread(
                target=self._run_estimator,
                name="PushNav-IMU-Estimator",
                daemon=True,
            )
            self._worker_thread.start()

            logger.info(
                "IMU reader and estimator started on %s at 115200 baud.",
                self.port,
            )
            return True

        except Exception:
            logger.exception("IMU startup failed; continuing without IMU.")
            self.stop()
            self.stream = None
            self.estimator = None
            return False

    def calibrate_stationary_gyro(
        self,
        *,
        minimum_samples: int = 200,
        maximum_samples: int = 500,
        maximum_sample_age_s: float = 0.5,
        maximum_window_s: float = 6.0,
    ) -> Calibration:
        """Calibrate gyro bias from recent stationary samples and save atomically.

        Existing calibration remains untouched if collection, validation, or
        persistence fails. The running estimator receives the new calibration
        only after the file has been successfully replaced.
        """
        stream = self.stream
        if stream is None or self.estimator is None:
            raise RuntimeError("IMU is not running")

        now = time.monotonic()
        records = stream.history.snapshot()
        records = [
            record for record in records
            if 0.0 <= now - record.received_at_monotonic <= maximum_window_s
        ]

        if not records:
            raise RuntimeError("No recent IMU samples; start the IMU and retry")

        latest_age = now - records[-1].received_at_monotonic
        if latest_age > maximum_sample_age_s:
            raise RuntimeError(
                f"IMU samples are stale ({latest_age:.2f}s); check the sensor connection"
            )

        records = records[-maximum_samples:]
        samples = [record.sample for record in records]

        calibration = calibrate_stationary(
            samples,
            minimum_samples=minimum_samples,
        )

        self.config_dir.mkdir(parents=True, exist_ok=True)
        calibration_path = self.config_dir / "imu_calibration.json"
        temp_path = None

        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self.config_dir,
                prefix="imu_calibration_",
                suffix=".tmp",
                delete=False,
            ) as temp_file:
                temp_path = Path(temp_file.name)
                json.dump(calibration.to_dict(), temp_file, indent=2)
                temp_file.write("\n")
                temp_file.flush()
                os.fsync(temp_file.fileno())

            os.replace(temp_path, calibration_path)
            temp_path = None
        finally:
            if temp_path is not None:
                try:
                    temp_path.unlink(missing_ok=True)
                except OSError:
                    logger.warning("Could not remove temporary calibration file")

        # Swap the immutable calibration object after durable persistence.
        # This does not change sensor alignment or reset the orientation.
        self.estimator.cal = calibration
        logger.info(
            "Stationary IMU gyro calibration saved (%d samples)",
            calibration.sample_count,
        )
        return calibration

    def _run_estimator(self) -> None:
        """Process records once; a >250 ms gap starts a new attitude epoch."""
        last_device_time_us: int | None = None
        try:
            while not self._worker_stop_event.is_set():
                if self.stream is None or self.estimator is None:
                    break
                records = self.stream.history.snapshot()
                for record in records:
                    if self._worker_stop_event.is_set():
                        break
                    if last_device_time_us is not None:
                        if record.device_time_us <= last_device_time_us:
                            continue
                        gap_us = record.device_time_us - last_device_time_us
                        if gap_us > self._max_contiguous_gap_us:
                            with self._estimator_lock:
                                self.estimator.reset_reference()
                            self._continuity_id += 1
                            with self._attitude_lock:
                                self._latest_attitude = None
                                self._latest_snapshot = None
                                self._orientation_history.clear()
                            logger.warning(
                                "IMU continuity lost (%.3fs gap); requiring a new "
                                "gravity reference and plate-solve anchor",
                                gap_us / 1_000_000.0,
                            )
                    with self._estimator_lock:
                        attitude = self.estimator.update(
                            record.sample, device_time_us=record.device_time_us
                        )
                        quaternion = tuple(self.estimator.q)
                    last_device_time_us = record.device_time_us
                    published = replace(
                        attitude,
                        sample_age_s=max(
                            0.0, time.monotonic() - record.received_at_monotonic
                        ),
                    )
                    snapshot = ImuSnapshot(
                        attitude=published,
                        quaternion=quaternion,
                        device_time_us=record.device_time_us,
                        received_at_monotonic=record.received_at_monotonic,
                        raw_sample=record.sample,
                        continuity_id=self._continuity_id,
                    )
                    with self._attitude_lock:
                        self._latest_attitude = published
                        self._latest_snapshot = snapshot
                        self._orientation_history.append(snapshot)
                time.sleep(0.005)
        except Exception:
            logger.exception("IMU estimator worker failed.")

    def stop(self) -> None:
        """Stop serial ingestion and then the estimator worker."""
        if self.stream is not None:
            self.stream.stop(join_timeout=2.0)

        self._worker_stop_event.set()

        if self._worker_thread is not None:
            self._worker_thread.join(timeout=2.0)
            if not self._worker_thread.is_alive():
                self._worker_thread = None

        if self.stream is not None:
            logger.info("IMU reader stopped.")
            self.stream = None
