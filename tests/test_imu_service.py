"""Tests for synchronized IMU service snapshots."""

from evf.imu.estimator import Attitude
from evf.imu.protocol import RawSample
from evf.imu.service import ImuService, ImuSnapshot


def test_latest_snapshot_is_none_before_first_sample(tmp_path):
    service = ImuService(tmp_path)

    assert service.latest_snapshot is None


def test_latest_snapshot_returns_data_and_safe_attitude_copy(tmp_path):
    service = ImuService(tmp_path)

    attitude = Attitude(
        roll_deg=1.0,
        pitch_deg=2.0,
        yaw_relative_deg=3.0,
        altitude_relative_deg=2.0,
        azimuth_relative_deg=3.0,
        sample_age_s=0.01,
        reference_initialized=True,
    )

    expected_quaternion = (1.0, 0.0, 0.0, 0.0)

    service._latest_snapshot = ImuSnapshot(
        attitude=attitude,
        quaternion=expected_quaternion,
        device_time_us=123456,
        received_at_monotonic=9876.5,
        raw_sample=RawSample(1, 123456, 0, 0, 16384, 0, 0, 131, 0),
    )

    snapshot = service.latest_snapshot

    assert snapshot is not None
    assert snapshot.quaternion == expected_quaternion
    assert snapshot.device_time_us == 123456
    assert snapshot.received_at_monotonic == 9876.5
    assert snapshot.attitude.roll_deg == 1.0

    assert snapshot is not service._latest_snapshot
    assert snapshot.attitude is not service._latest_snapshot.attitude

    snapshot.attitude.roll_deg = 99.0

    assert service.latest_snapshot.attitude.roll_deg == 1.0

def test_worker_publishes_synchronized_snapshot(tmp_path, monkeypatch):
    from evf.imu.stream import ImuSampleRecord
    from evf.imu.protocol import RawSample

    service = ImuService(tmp_path)

    sample = RawSample(
        1, 100000,
        0, 0, 16384,
        0, 0, 131,
        0,
    )

    record = ImuSampleRecord(
        sample=sample,
        received_at_monotonic=10.0,
        batch_id=1,
        device_time_us=100000,
    )

    expected_attitude = Attitude(
        roll_deg=1.0,
        pitch_deg=2.0,
        yaw_relative_deg=3.0,
        altitude_relative_deg=2.0,
        azimuth_relative_deg=3.0,
        reference_initialized=True,
    )

    expected_quaternion = (0.9238795, 0.0, 0.3826834, 0.0)

    class FakeHistory:
        def snapshot(self):
            return (record,)

    class FakeStream:
        history = FakeHistory()

    class FakeEstimator:
        q = expected_quaternion

        def update(self, received_sample, device_time_us):
            assert received_sample is sample
            assert device_time_us == 100000
            service._worker_stop_event.set()
            return expected_attitude

    service.stream = FakeStream()
    service.estimator = FakeEstimator()

    monkeypatch.setattr(
        "evf.imu.service.time.monotonic",
        lambda: 10.025,
    )

    service._run_estimator()

    snapshot = service.latest_snapshot

    assert snapshot is not None
    assert snapshot.attitude.roll_deg == 1.0
    assert snapshot.quaternion == expected_quaternion
    assert snapshot.device_time_us == 100000
    assert snapshot.received_at_monotonic == 10.0
    assert abs(snapshot.attitude.sample_age_s - 0.025) < 1e-9
    assert service.latest_attitude.roll_deg == 1.0



def _make_interpolation_snapshot(timestamp, quaternion):
    attitude = Attitude(
        roll_deg=0.0,
        pitch_deg=0.0,
        yaw_relative_deg=0.0,
        altitude_relative_deg=0.0,
        azimuth_relative_deg=0.0,
        reference_initialized=True,
    )
    sample = RawSample(1, timestamp, 0, 0, 16384, 0, 0, 131, 0)
    return ImuSnapshot(
        attitude=attitude,
        quaternion=quaternion,
        device_time_us=timestamp,
        received_at_monotonic=1.0,
        raw_sample=sample,
    )


def test_service_interpolates_orientation_from_history(tmp_path):
    import math

    service = ImuService(tmp_path)
    start = (1.0, 0.0, 0.0, 0.0)
    end = (math.cos(math.pi / 4), 0.0, 0.0, math.sin(math.pi / 4))
    service._orientation_history.extend([
        _make_interpolation_snapshot(10000, start),
        _make_interpolation_snapshot(20000, end),
    ])

    actual = service.interpolated_quaternion(15000)
    expected = (math.cos(math.pi / 8), 0.0, 0.0, math.sin(math.pi / 8))

    assert actual is not None
    for a, b in zip(actual, expected):
        assert abs(a - b) < 1e-9


def test_service_interpolation_rejects_target_outside_history(tmp_path):
    service = ImuService(tmp_path)
    q = (1.0, 0.0, 0.0, 0.0)
    service._orientation_history.extend([
        _make_interpolation_snapshot(10000, q),
        _make_interpolation_snapshot(20000, q),
    ])

    assert service.interpolated_quaternion(9000) is None
    assert service.interpolated_quaternion(21000) is None


def test_service_interpolation_rejects_large_history_gap(tmp_path):
    service = ImuService(tmp_path)
    q = (1.0, 0.0, 0.0, 0.0)
    service._orientation_history.extend([
        _make_interpolation_snapshot(10000, q),
        _make_interpolation_snapshot(50000, q),
    ])

    assert service.interpolated_quaternion(30000) is None

def test_orientation_at_host_time_maps_and_interpolates(tmp_path):
    import math
    from types import SimpleNamespace

    service = ImuService(tmp_path)

    start = (1.0, 0.0, 0.0, 0.0)
    end = (
        math.cos(math.pi / 4),
        0.0,
        0.0,
        math.sin(math.pi / 4),
    )

    service._orientation_history.extend([
        _make_interpolation_snapshot(10000, start),
        _make_interpolation_snapshot(20000, end),
    ])

    records = [
        SimpleNamespace(
            batch_id=1,
            device_time_us=10000,
            received_at_monotonic=100.010,
        ),
        SimpleNamespace(
            batch_id=2,
            device_time_us=20000,
            received_at_monotonic=100.020,
        ),
    ]

    class FakeHistory:
        def snapshot(self):
            return tuple(records)

    service.stream = SimpleNamespace(history=FakeHistory())

    result = service.orientation_at_host_time(100.015)

    assert result is not None
    device_time_us, quaternion = result
    assert device_time_us == 15000

    expected = (
        math.cos(math.pi / 8),
        0.0,
        0.0,
        math.sin(math.pi / 8),
    )
    for actual, wanted in zip(quaternion, expected):
        assert abs(actual - wanted) < 1e-9


def test_orientation_at_host_time_returns_none_without_stream(tmp_path):
    service = ImuService(tmp_path)

    assert service.orientation_at_host_time(100.0) is None


def test_orientation_at_host_time_returns_none_without_bracketing_samples(
    tmp_path,
):
    from types import SimpleNamespace

    service = ImuService(tmp_path)

    q = (1.0, 0.0, 0.0, 0.0)
    service._orientation_history.extend([
        _make_interpolation_snapshot(10000, q),
        _make_interpolation_snapshot(20000, q),
    ])

    class FakeHistory:
        def snapshot(self):
            return (
                SimpleNamespace(
                    batch_id=1,
                    device_time_us=10000,
                    received_at_monotonic=100.010,
                ),
                SimpleNamespace(
                    batch_id=2,
                    device_time_us=20000,
                    received_at_monotonic=100.020,
                ),
            )

    service.stream = SimpleNamespace(history=FakeHistory())

    assert service.orientation_at_host_time(100.030) is None

def _stationary_calibration_service(tmp_path, monkeypatch, *, count=250, age=0.1):
    import time
    from types import SimpleNamespace

    service = ImuService(tmp_path)

    # Constant, stationary samples: 1 g on Z, zero gyro on all axes.
    sample = RawSample(1, 100000, 0, 0, 16384, 0, 0, 0, 0)
    now = 100.0
    records = [
        SimpleNamespace(
            sample=sample,
            received_at_monotonic=now - age,
            batch_id=index,
            device_time_us=100000 + index,
        )
        for index in range(count)
    ]

    class FakeHistory:
        def snapshot(self):
            return tuple(records)

    class FakeEstimator:
        cal = None

    service.stream = SimpleNamespace(history=FakeHistory())
    service.estimator = FakeEstimator()
    monkeypatch.setattr("evf.imu.service.time.monotonic", lambda: now)
    return service


def test_stationary_calibration_saves_and_updates_estimator(tmp_path, monkeypatch):
    import json

    service = _stationary_calibration_service(tmp_path, monkeypatch)

    calibration = service.calibrate_stationary_gyro()

    saved_path = tmp_path / "imu_calibration.json"
    assert saved_path.exists()
    assert json.loads(saved_path.read_text(encoding="utf-8")) == calibration.to_dict()
    assert service.estimator.cal == calibration
    assert calibration.sample_count == 250
    assert not list(tmp_path.glob("imu_calibration_*.tmp"))


def test_stationary_calibration_rejects_insufficient_samples_without_overwrite(
    tmp_path, monkeypatch
):
    import json

    path = tmp_path / "imu_calibration.json"
    original = {"gyro_bias_dps": [1.0, 2.0, 3.0], "existing": True}
    path.write_text(json.dumps(original), encoding="utf-8")

    service = _stationary_calibration_service(
        tmp_path, monkeypatch, count=199
    )

    import pytest
    with pytest.raises(ValueError):
        service.calibrate_stationary_gyro()

    assert json.loads(path.read_text(encoding="utf-8")) == original
    assert service.estimator.cal is None


def test_stationary_calibration_rejects_stale_samples_without_overwrite(
    tmp_path, monkeypatch
):
    import json
    import pytest

    path = tmp_path / "imu_calibration.json"
    original = {"preserve": "previous calibration"}
    path.write_text(json.dumps(original), encoding="utf-8")

    service = _stationary_calibration_service(
        tmp_path, monkeypatch, count=250, age=1.0
    )

    with pytest.raises(RuntimeError, match="stale"):
        service.calibrate_stationary_gyro()

    assert json.loads(path.read_text(encoding="utf-8")) == original
    assert service.estimator.cal is None


def test_stationary_calibration_replace_failure_preserves_old_file(
    tmp_path, monkeypatch
):
    import json
    import pytest

    path = tmp_path / "imu_calibration.json"
    original = {"preserve": "old"}
    path.write_text(json.dumps(original), encoding="utf-8")

    service = _stationary_calibration_service(tmp_path, monkeypatch)

    def fail_replace(source, destination):
        raise OSError("simulated replace failure")

    monkeypatch.setattr("evf.imu.service.os.replace", fail_replace)

    with pytest.raises(OSError, match="simulated replace failure"):
        service.calibrate_stationary_gyro()

    assert json.loads(path.read_text(encoding="utf-8")) == original
    assert service.estimator.cal is None
    assert not list(tmp_path.glob("imu_calibration_*.tmp"))


def test_set_sensor_to_telescope_resets_published_orientation_state(tmp_path):
    from unittest.mock import MagicMock

    service = ImuService(tmp_path)
    estimator = MagicMock()
    service.estimator = estimator
    service._latest_attitude = Attitude(
        roll_deg=1.0,
        pitch_deg=2.0,
        yaw_relative_deg=3.0,
        altitude_relative_deg=2.0,
        azimuth_relative_deg=3.0,
        reference_initialized=True,
    )
    service._latest_snapshot = _make_interpolation_snapshot(
        10000, (1.0, 0.0, 0.0, 0.0)
    )
    service._orientation_history.append(service._latest_snapshot)

    matrix = [
        [0.0, -1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
    service.set_sensor_to_telescope(matrix)

    estimator.set_sensor_to_telescope.assert_called_once_with(matrix)
    assert service.sensor_mapping_verified is True
    assert service.latest_attitude is None
    assert service.latest_snapshot is None
    assert len(service.orientation_history) == 0

