"""Regression coverage for signed physical IMU sweep integration."""

from types import SimpleNamespace
import json

import pytest

from evf.imu.alignment_workflow import ImuAlignmentWorkflow


class _History:
    def __init__(self, records):
        self.records = records

    def samples_between(self, start, end):
        return self.records

    def latest(self):
        return None


class _Stream:
    def __init__(self, records):
        self.history = _History(records)


def test_sweep_integrates_full_timeline_and_keeps_requested_sign(monkeypatch, tmp_path):
    # 200 samples at 100 Hz: 20 stationary, 160 moving at 20 dps,
    # 20 stationary. A 1.6 s sweep should integrate to ~32 degrees.
    records = []
    for i in range(200):
        rate = 20.0 if 20 <= i < 180 else 0.0
        sample = SimpleNamespace(gyro_dps=(rate, 0.0, 0.0))
        records.append(SimpleNamespace(
            device_time_us=i * 10_000,
            received_at_monotonic=10.0 + i * 0.01,
            sample=sample,
        ))

    (tmp_path / "imu_calibration.json").write_text(json.dumps({"gyro_bias_dps": [0, 0, 0]}), encoding="utf-8")
    service = SimpleNamespace(stream=_Stream(records), config_dir=tmp_path)
    workflow = ImuAlignmentWorkflow(service)
    clock = iter((10.0, 12.0))
    monkeypatch.setattr("evf.imu.alignment_workflow.time.monotonic", lambda: next(clock))

    workflow.start_sensor_sweep("altitude", 1)
    result = workflow.finish_sensor_sweep()

    sweep = result["sensor_sweeps"][0]
    assert sweep["sample_count"] == 200
    assert sweep["integrated_angle_deg"] == pytest.approx(32.0, abs=0.2)
    assert sweep["gyro_noise_dps"] < 0.01
    assert sweep["axis"] == "altitude"
