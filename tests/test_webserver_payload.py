"""Tests for /ws JSON payload schema."""

import json

from evf.config.manager import ConfigManager
from evf.engine.frame_buffer import LatestFrame
from evf.engine.goto_target import GotoTarget
from evf.engine.pointing import PointingState
from evf.engine.state import StateMachine
from evf.webserver.server import WebServer


def test_payload_contains_new_fields(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    ws = WebServer(
        PointingState(), StateMachine(), GotoTarget(), ConfigManager(config_dir=tmp_path / "evf-config"),
        frame_buffer=LatestFrame(),
        camera_controls=lambda: [
            {"id": "exposure", "label": "Exposure", "min": 0, "max": 100, "step": 1, "value": 50, "unit": "ms"},
        ],
        sync_state=lambda: {
            "in_progress": False, "candidates": [], "selected_idx": None, "error": None,
        },
        activity=lambda: {
            "stellarium": {"active": False, "address": "localhost:10001"},
            "lx200":      {"active": False, "address": "0.0.0.0:4030"},
            "webserver":  {"url": "http://192.168.1.42:8765"},
            "audio_enabled": True,
        },
    )
    payload = ws._build_payload()
    # Existing fields still present
    assert "state" in payload
    assert "pointing" in payload
    assert payload["imu"] is None
    # New fields
    assert "controls" in payload and isinstance(payload["controls"], list)
    assert "sync" in payload and "in_progress" in payload["sync"]
    assert "stellarium" in payload and "active" in payload["stellarium"]
    assert "lx200" in payload
    assert "webserver" in payload
    assert "audio_enabled" in payload
    assert "camera" in payload  # centroid arrays
    # Roundtrip JSON-serializable
    assert json.dumps(payload)


def test_payload_contains_location_field(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    cfg = ConfigManager(config_dir=tmp_path / "evf-config")
    ws = WebServer(
        PointingState(), StateMachine(), GotoTarget(), cfg,
        frame_buffer=LatestFrame(),
        location=lambda: {
            "latitude": 13.0878,
            "longitude": 80.2785,
            "source": "manual",
        },
    )
    payload = ws._build_payload()
    assert "location" in payload
    assert payload["location"]["source"] == "manual"
    assert payload["location"]["latitude"] == 13.0878


def test_payload_contains_live_imu_attitude(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    cfg = ConfigManager(config_dir=tmp_path / "evf-config")

    attitude = type("TestAttitude", (), {
        "altitude_relative_deg": 12.5,
        "azimuth_relative_deg": -4.25,
        "roll_deg": 1.5,
        "pitch_deg": -2.0,
        "yaw_relative_deg": 3.75,
        "yaw_drift_warning": True,
        "sample_age_s": 0.02,
        "accel_correction_used": True,
        "reference_initialized": True,
    })()

    ws = WebServer(
        PointingState(), StateMachine(), GotoTarget(), cfg,
        frame_buffer=LatestFrame(),
        imu_attitude=lambda: attitude,
    )

    payload = ws._build_payload()
    imu = payload["imu"]

    assert imu is not None
    assert imu["available"] is True
    assert imu["altitude_relative_deg"] == 12.5
    assert imu["azimuth_relative_deg"] == -4.25
    assert imu["roll_deg"] == 1.5
    assert imu["pitch_deg"] == -2.0
    assert imu["yaw_relative_deg"] == 3.75
    assert imu["yaw_drift_warning"] is True
    assert imu["sample_age_s"] == 0.02
    assert imu["accel_correction_used"] is True
    assert imu["reference_initialized"] is True

    assert "pointing" in payload
    assert json.dumps(payload)
