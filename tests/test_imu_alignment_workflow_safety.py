import json
import math
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from evf.imu.alignment_workflow import ImuAlignmentWorkflow, SensorSweep
from evf.imu.alignment_calibration import fit_rotation
import evf.engine.engine as engine_module


class FakeHistory:
    def __init__(self):
        self.records = []

    def latest(self):
        return self.records[-1] if self.records else None

    def samples_between(self, start, end):
        return list(self.records)


class FakeService:
    def __init__(self, config_dir):
        self.config_dir = config_dir
        self.stream = SimpleNamespace(history=FakeHistory())
        self.orientations = {}

    def orientation_at_host_time(self, timestamp):
        return self.orientations.get(timestamp)


def make_records(axis, rate=10.0, wander=False):
    records = []
    axis = np.asarray(axis, dtype=float)
    perpendicular = np.cross(axis, [0.0, 0.0, 1.0])
    if np.linalg.norm(perpendicular) < 0.1:
        perpendicular = np.cross(axis, [0.0, 1.0, 0.0])
    perpendicular /= np.linalg.norm(perpendicular)

    for i in range(160):
        if 16 <= i < 144:
            if wander and i >= 80:
                gyro = perpendicular * rate
            else:
                gyro = axis * rate
        else:
            gyro = np.zeros(3)

        records.append(SimpleNamespace(
            device_time_us=i * 10_000,
            sample=SimpleNamespace(
                gyro_dps=tuple(float(v) for v in gyro)
            ),
        ))
    return records


def make_workflow(tmp_path, monkeypatch):
    service = FakeService(tmp_path)
    (tmp_path / "imu_calibration.json").write_text(
        json.dumps({
            "gyro_bias_dps": [0.0, 0.0, 0.0],
            "sample_count": 300,
        }),
        encoding="utf-8",
    )

    workflow = ImuAlignmentWorkflow(service)
    clock = [100.0]

    def fake_monotonic():
        value = clock[0]
        clock[0] += 1.5
        return value

    monkeypatch.setattr(
        "evf.imu.alignment_workflow.time.monotonic",
        fake_monotonic,
    )
    return workflow, service


def capture_sweep(workflow, service, axis, direction, records):
    service.stream.history.records = records
    workflow.start_sensor_sweep(axis, direction)
    return workflow.finish_sensor_sweep()


def test_reverse_sweep_direction_is_rejected(tmp_path, monkeypatch):
    workflow, service = make_workflow(tmp_path, monkeypatch)

    # Altitude positive is deliberately moved in the negative direction.
    capture_sweep(
        workflow, service, "altitude", 1,
        make_records([1, 0, 0], rate=-10.0),
    )
    capture_sweep(
        workflow, service, "altitude", -1,
        make_records([1, 0, 0], rate=-10.0),
    )

    # The azimuth pair is correctly performed in opposite directions.
    capture_sweep(
        workflow, service, "azimuth", 1,
        make_records([0, 1, 0], rate=10.0),
    )
    capture_sweep(
        workflow, service, "azimuth", -1,
        make_records([0, 1, 0], rate=-10.0),
    )

    with pytest.raises(ValueError, match="sweeps disagree"):
        workflow.fit_sensor_alignment()


def test_wandering_sweep_axis_is_rejected(tmp_path, monkeypatch):
    workflow, service = make_workflow(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="noise|inconsistent|reversed"):
        capture_sweep(
            workflow, service, "altitude", 1,
            make_records([1, 0, 0], wander=True),
        )


def test_insufficient_integrated_movement_is_rejected(tmp_path, monkeypatch):
    workflow, service = make_workflow(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="moved only"):
        capture_sweep(
            workflow, service, "altitude", 1,
            make_records([1, 0, 0], rate=2.0),
        )


def test_collinear_calibration_geometry_is_rejected():
    with pytest.raises(ValueError, match="angular diversity"):
        fit_rotation(
            [(1, 0, 0), (1, 0, 0), (-1, 0, 0), (-1, 0, 0)],
            [(0, 1, 0), (0, 1, 0), (0, 0, 1), (0, 0, 1)],
        )


def test_excessive_sensor_fit_residual_is_rejected():
    workflow = ImuAlignmentWorkflow(None)

    def sweep(axis, direction, vector):
        return SensorSweep(
            axis=axis,
            direction=direction,
            sensor_axis=tuple(vector),
            sample_count=160,
            duration_s=1.5,
            integrated_angle_deg=12.8,
            gyro_noise_dps=0.0,
            started_at=0.0,
            finished_at=1.5,
        )

    workflow._sweeps = [
        sweep("altitude", 1, (1, 0, 0)),
        sweep("altitude", -1, (1, 0, 0)),
        sweep("azimuth", 1, (0, 1, 0)),
        sweep("azimuth", -1, (0, 0, 1)),
    ]

    with pytest.raises(ValueError, match="residual|alignment"):
        workflow.fit_sensor_alignment()


def test_missing_frame_time_imu_orientation_is_rejected(tmp_path):
    workflow = ImuAlignmentWorkflow(FakeService(tmp_path))
    pointing = SimpleNamespace(
        ra_j2000=10.0,
        dec_j2000=20.0,
        roll=0.0,
    )

    metadata = SimpleNamespace(
        timestamp_quality="callback", clock_sync_verified=True,
        timing_uncertainty_s=None,
    )
    with pytest.raises(ValueError, match="orientation"):
        workflow.capture_camera_pair(pointing, 123.0, 1, metadata)


def test_camera_fit_rejects_repeated_identical_attitudes(tmp_path):
    service = FakeService(tmp_path)
    workflow = ImuAlignmentWorkflow(service)
    identity_quaternion = (1.0, 0.0, 0.0, 0.0)

    pointing = SimpleNamespace(
        ra_j2000=10.0,
        dec_j2000=20.0,
        roll=0.0,
    )

    for frame_id in range(1, 5):
        timestamp = float(frame_id)
        service.orientations[timestamp] = (
            frame_id * 1_000_000,
            identity_quaternion,
        )
        metadata = SimpleNamespace(
            timestamp_quality="callback", clock_sync_verified=True,
            timing_uncertainty_s=None,
        )
        workflow.capture_camera_pair(pointing, timestamp, frame_id, metadata)

    with pytest.raises(ValueError, match="coverage|attitude"):
        workflow.fit_camera_alignment()


def test_finalization_rejects_incomplete_or_unverified_calibration(tmp_path):
    engine_type = next(
        cls for cls in vars(engine_module).values()
        if isinstance(cls, type)
        and hasattr(cls, "finalize_imu_alignment")
    )

    engine = engine_type.__new__(engine_type)
    engine._config = SimpleNamespace(path=tmp_path / "config.json")
    engine._imu_alignment_workflow = ImuAlignmentWorkflow(None)
    engine._imu_service = SimpleNamespace(sensor_mapping_verified=True)

    with pytest.raises(ValueError, match="fits must pass"):
        engine.finalize_imu_alignment()

    assert not (tmp_path / "imu" / "sensor_alignment.json").exists()

    valid_measured = {
        "sensor_to_telescope": np.eye(3).tolist(),
        "camera_from_telescope": np.eye(3).tolist(),
        "calibration_method": "synthetic_test_only",
        "camera_timing_evidence": {
            "timestamp_source": "directshow_sample_callback_qpc",
            "clock_sync_verified": True,
            "exposure_midpoint_verified": False,
        },
    }
    engine._imu_alignment_workflow = SimpleNamespace(
        measured_result=lambda: valid_measured
    )
    engine._imu_service.sensor_mapping_verified = False

    with pytest.raises(RuntimeError, match="not applied and verified"):
        engine.finalize_imu_alignment()

    assert not (tmp_path / "imu" / "sensor_alignment.json").exists()


def test_finalization_persists_and_reload_validates_transforms(tmp_path):
    engine_type = next(
        cls for cls in vars(engine_module).values()
        if isinstance(cls, type)
        and hasattr(cls, "finalize_imu_alignment")
    )

    engine = engine_type.__new__(engine_type)
    engine._config = SimpleNamespace(path=tmp_path / "config.json")
    engine._imu_service = SimpleNamespace(sensor_mapping_verified=True)

    measured = {
        "sensor_to_telescope": np.eye(3).tolist(),
        "camera_from_telescope": np.eye(3).tolist(),
        "calibration_method": "synthetic_test_only",
        "camera_timing_evidence": {
            "timestamp_source": "directshow_sample_callback_qpc",
            "clock_sync_verified": True,
            "exposure_midpoint_verified": False,
        },
    }
    engine._imu_alignment_workflow = SimpleNamespace(
        measured_result=lambda: measured
    )
    engine._imu_camera_from_telescope = None
    engine._imu_camera_capture_active = True
    engine._plate_solve_anchor_lock = threading.Lock()
    engine._plate_solve_anchor = object()

    result = engine.finalize_imu_alignment()
    assert result["saved"] is True

    alignment_path = tmp_path / "imu" / "sensor_alignment.json"
    assert alignment_path.exists()

    saved = json.loads(alignment_path.read_text(encoding="utf-8"))
    assert saved["mapping_status"] == "VERIFIED"
    assert saved["camera_from_telescope_status"] == "VERIFIED"
    assert saved["calibration_evidence"]["calibration_method"] == (
        "synthetic_test_only"
    )

    # Simulate a fresh engine loading the persisted transform.
    loaded = engine_type._load_imu_camera_alignment(tmp_path / "imu")
    np.testing.assert_allclose(loaded, np.eye(3), atol=1e-8)

    assert engine._imu_camera_capture_active is False
    assert engine._plate_solve_anchor is None

