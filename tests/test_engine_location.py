def _telescope_to_sky(ra, dec, roll=0.0):
    """Synthetic telescope-body (+X boresight) to celestial transform."""
    import numpy as np
    from evf.solver.sync import orientation_from_radec_roll

    camera_to_sky = np.asarray(
        orientation_from_radec_roll(ra, dec, roll), dtype=float
    )
    camera_from_telescope = np.asarray(
        (
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (1.0, 0.0, 0.0),
        ),
        dtype=float,
    )
    result = camera_to_sky @ camera_from_telescope
    return tuple(tuple(float(value) for value in row) for row in result)

"""Tests for engine location resolution + setter."""

from unittest.mock import MagicMock
from types import SimpleNamespace
import time

import pytest

import numpy as np

from evf.config.manager import ConfigManager
from evf.engine.engine import Engine


def _make_engine(tmp_path):
    """Build an Engine with config and test alignment isolated to tmp_path."""
    cfg = ConfigManager(config_dir=tmp_path / "evf-config")
    eng = Engine(dev_mode=False, config=cfg)
    # Test-only identity calibration: telescope and camera body axes coincide.
    eng._imu_camera_from_telescope = np.eye(3)
    return eng


def test_startup_imu_uses_configured_serial_port(tmp_path, monkeypatch):
    eng = _make_engine(tmp_path)
    eng._config.imu_port = "COM7"

    service = MagicMock()
    service_class = MagicMock(return_value=service)
    monkeypatch.setattr("evf.engine.engine.ImuService", service_class)

    eng.startup_imu()

    service_class.assert_called_once_with(
        eng._config.path.parent / "imu",
        port="COM7",
    )
    service.start.assert_called_once()

def _sky_rotation(ra, dec, roll=0.0):
    from evf.solver.sync import orientation_from_radec_roll
    matrix = orientation_from_radec_roll(ra, dec, roll)
    return tuple(tuple(float(value) for value in row) for row in matrix)


def test_location_none_by_default(tmp_path):
    eng = _make_engine(tmp_path)
    assert eng.location == {"latitude": None, "longitude": None, "source": None}


def test_location_from_manual_config(tmp_path):
    eng = _make_engine(tmp_path)
    eng.set_location(13.0878, 80.2785)
    assert eng.location == {
        "latitude": 13.0878,
        "longitude": 80.2785,
        "source": "manual",
    }


def test_location_from_stellarium_takes_precedence(tmp_path):
    eng = _make_engine(tmp_path)
    eng.set_location(13.0878, 80.2785)  # manual

    # Mock a Stellarium server with a location
    fake_server = MagicMock()
    fake_server.stellarium_status = {
        "location": {"name": "London", "latitude": 51.5, "longitude": -0.1},
    }
    eng._stellarium = fake_server

    loc = eng.location
    assert loc["source"] == "stellarium"
    assert loc["latitude"] == 51.5
    assert loc["longitude"] == -0.1


def test_set_location_clear(tmp_path):
    eng = _make_engine(tmp_path)
    eng.set_location(13.0878, 80.2785)
    eng.set_location(None, None)
    assert eng.location == {"latitude": None, "longitude": None, "source": None}


def test_imu_snapshot_none_before_service_start(tmp_path):
    eng = _make_engine(tmp_path)
    assert eng.imu_snapshot is None


def test_imu_snapshot_returns_latest_service_snapshot(tmp_path):
    eng = _make_engine(tmp_path)
    expected_snapshot = object()

    fake_service = MagicMock()
    fake_service.latest_snapshot = expected_snapshot
    eng._imu_service = fake_service

    assert eng.imu_snapshot is expected_snapshot


def _anchor_test_pointing(valid=True):
    return SimpleNamespace(
        ra_j2000=120.5,
        dec_j2000=22.25,
        roll=3.0,
        last_success_timestamp=time.monotonic(),
        valid=valid,
    )


def _anchor_test_service(orientation=(123456, (1.0, 0.0, 0.0, 0.0))):
    service = MagicMock()
    service.orientation_at_host_time.return_value = orientation
    return service


def test_plate_solve_anchor_uses_orientation_at_frame_timestamp(tmp_path):
    eng = _make_engine(tmp_path)
    service = _anchor_test_service()
    eng._imu_service = service

    # Simulate a solver callback arriving well after the camera frame.
    frame_timestamp = time.monotonic() - 2.0
    frame_id = 42

    eng._capture_plate_solve_anchor(
        _anchor_test_pointing(), frame_timestamp, frame_id
    )

    service.orientation_at_host_time.assert_called_once_with(frame_timestamp)

    anchor = eng.plate_solve_anchor
    assert anchor is not None
    assert anchor.ra_j2000 == 120.5
    assert anchor.dec_j2000 == 22.25
    assert anchor.roll == 3.0
    assert anchor.frame_timestamp == frame_timestamp
    assert anchor.frame_id == frame_id
    assert anchor.quaternion == (1.0, 0.0, 0.0, 0.0)
    assert anchor.imu_device_time_us == 123456


def test_plate_solve_anchor_clears_when_imu_service_missing(tmp_path):
    eng = _make_engine(tmp_path)
    eng._imu_service = _anchor_test_service()
    eng._capture_plate_solve_anchor(_anchor_test_pointing(), 100.0, 1)
    assert eng.plate_solve_anchor is not None

    eng._imu_service = None
    eng._capture_plate_solve_anchor(_anchor_test_pointing(), 101.0, 2)

    assert eng.plate_solve_anchor is None


def test_plate_solve_anchor_clears_when_frame_orientation_unavailable(tmp_path):
    eng = _make_engine(tmp_path)
    service = _anchor_test_service(orientation=None)
    eng._imu_service = service

    eng._capture_plate_solve_anchor(_anchor_test_pointing(), 100.0, 1)

    service.orientation_at_host_time.assert_called_once_with(100.0)
    assert eng.plate_solve_anchor is None


def test_plate_solve_anchor_does_not_request_orientation_for_invalid_pointing(tmp_path):
    eng = _make_engine(tmp_path)
    service = _anchor_test_service()
    eng._imu_service = service

    eng._capture_plate_solve_anchor(
        _anchor_test_pointing(valid=False), 100.0, 1
    )

    service.orientation_at_host_time.assert_not_called()
    assert eng.plate_solve_anchor is None


def test_imu_quaternion_transform_batch(tmp_path):
    import math

    eng = _make_engine(tmp_path)
    h = math.sqrt(0.5)
    cases = [
        ((1.0, 0.0, 0.0, 0.0), (0.2, -0.3, 0.4), (0.2, -0.3, 0.4)),
        ((h, h, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        ((h, 0.0, h, 0.0), (0.0, 0.0, 1.0), (1.0, 0.0, 0.0)),
        ((h, 0.0, 0.0, h), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
    ]
    for q, vector, expected in cases:
        actual = eng._imu_rotate_vector(q, vector)
        assert all(math.isclose(a, e, abs_tol=1e-9)
                   for a, e in zip(actual, expected))


def test_imu_quaternion_inverse_rotation(tmp_path):
    import math

    eng = _make_engine(tmp_path)
    angle = math.pi / 2
    q = (math.cos(angle / 2), 0.0, 0.0, math.sin(angle / 2))
    inverse = (q[0], -q[1], -q[2], -q[3])
    rotated = eng._imu_rotate_vector(q, (1.0, 0.0, 0.0))
    restored = eng._imu_rotate_vector(inverse, rotated)
    assert all(math.isclose(a, e, abs_tol=1e-9)
               for a, e in zip(restored, (1.0, 0.0, 0.0)))


def test_imu_propagation_normalizes_quaternions(tmp_path):
    import math

    eng = _make_engine(tmp_path)
    angle = math.pi / 2
    q = (math.cos(angle / 2), 0.0, 0.0, math.sin(angle / 2))
    anchor = SimpleNamespace(
        quaternion=(2.0, 0.0, 0.0, 0.0),
        ra_j2000=0.0,
        dec_j2000=0.0,
        telescope_to_sky=_telescope_to_sky(0.0, 0.0),
    )
    result = eng._imu_propagate_radec(anchor, tuple(2.0 * v for v in q))
    assert math.isclose(result[0], 90.0, abs_tol=1e-8)
    assert math.isclose(result[1], 0.0, abs_tol=1e-8)


def test_imu_propagation_handles_ra_wraparound(tmp_path):
    import math

    eng = _make_engine(tmp_path)
    angle = math.radians(2.0)
    q = (math.cos(angle / 2), 0.0, 0.0, math.sin(angle / 2))
    anchor = SimpleNamespace(
        quaternion=(1.0, 0.0, 0.0, 0.0),
        ra_j2000=359.0,
        dec_j2000=0.0,
        telescope_to_sky=_telescope_to_sky(359.0, 0.0),
    )
    ra, dec = eng._imu_propagate_radec(anchor, q)
    assert math.isclose(ra, 1.0, abs_tol=1e-8)
    assert math.isclose(dec, 0.0, abs_tol=1e-8)


def test_imu_propagation_preserves_high_declination(tmp_path):
    import math

    eng = _make_engine(tmp_path)
    angle = math.radians(90.0)
    q = (math.cos(angle / 2), 0.0, 0.0, math.sin(angle / 2))
    anchor = SimpleNamespace(
        quaternion=(1.0, 0.0, 0.0, 0.0),
        ra_j2000=45.0,
        dec_j2000=89.999,
        telescope_to_sky=_telescope_to_sky(45.0, 89.999),
    )
    ra, dec = eng._imu_propagate_radec(anchor, q)
    assert math.isclose(ra, 135.0, abs_tol=1e-6)
    assert math.isclose(dec, 0.0, abs_tol=1e-6)
    assert -90.0 <= dec <= 90.0


def test_imu_propagation_rejects_zero_quaternion(tmp_path):
    eng = _make_engine(tmp_path)
    anchor = SimpleNamespace(
        quaternion=(1.0, 0.0, 0.0, 0.0),
        ra_j2000=120.0,
        dec_j2000=30.0,
        telescope_to_sky=_telescope_to_sky(120.0, 30.0),
    )
    assert eng._imu_propagate_radec(
        anchor, (0.0, 0.0, 0.0, 0.0)
    ) is None
    anchor.quaternion = (0.0, 0.0, 0.0, 0.0)
    assert eng._imu_propagate_radec(
        anchor, (1.0, 0.0, 0.0, 0.0)
    ) is None


def test_imu_pointing_update_preserves_metadata_and_rejects_stale_anchor(tmp_path):
    eng = _make_engine(tmp_path)
    pointing = eng.pointing_state
    pointing.update(
        ra_j2000=120.5,
        dec_j2000=22.25,
        roll=3.0,
        matches=18,
        prob=0.01,
        all_centroids=None,
        matched_centroids=None,
        image_size=None,
    )
    before = pointing.read()

    assert pointing.update_from_imu(
        121.0, 22.5, before.last_success_timestamp
    )
    after = pointing.read()
    assert after.ra_j2000 == 121.0
    assert after.dec_j2000 == 22.5
    assert after.roll == before.roll
    assert after.matches == before.matches
    assert after.prob == before.prob
    assert after.last_success_timestamp == before.last_success_timestamp

    pointing.update(
        ra_j2000=200.0,
        dec_j2000=-10.0,
        roll=7.0,
        matches=25,
        prob=0.02,
        all_centroids=None,
        matched_centroids=None,
        image_size=None,
    )
    newer = pointing.read()
    assert not pointing.update_from_imu(
        122.0, 23.0, before.last_success_timestamp
    )
    unchanged = pointing.read()
    assert unchanged.ra_j2000 == newer.ra_j2000
    assert unchanged.dec_j2000 == newer.dec_j2000
    assert unchanged.last_success_timestamp == newer.last_success_timestamp



class _OneIterationEvent:
    """Let the worker execute once, then stop on its next wait."""
    def __init__(self):
        self.calls = 0

    def wait(self, timeout):
        self.calls += 1
        return self.calls > 1


def _prepare_imu_worker_test(eng, sample_age=0.0):
    from types import SimpleNamespace

    eng.pointing_state.update(
        ra_j2000=0.0,
        dec_j2000=0.0,
        roll=0.0,
        matches=20,
        prob=0.01,
        all_centroids=None,
        matched_centroids=None,
        image_size=None,
    )
    solved = eng.pointing_state.read()
    eng._plate_solve_anchor = SimpleNamespace(
        ra_j2000=0.0,
        dec_j2000=0.0,
        roll=0.0,
        pointing_timestamp=solved.last_success_timestamp,
        frame_id=17,
        quaternion=(1.0, 0.0, 0.0, 0.0),
        telescope_to_sky=_telescope_to_sky(0.0, 0.0),
    )
    eng._imu_service = MagicMock()
    eng._imu_service.latest_snapshot = SimpleNamespace(
        quaternion=(2 ** -0.5, 0.0, 0.0, 2 ** -0.5),
        device_time_us=123456,
        received_at_monotonic=time.monotonic() - sample_age,
    )
    eng._imu_tracking_stop = _OneIterationEvent()


def test_imu_anchor_requires_verified_camera_alignment(tmp_path):
    eng = _make_engine(tmp_path)
    eng.pointing_state.update(
        ra_j2000=10.0,
        dec_j2000=20.0,
        roll=0.0,
        matches=10,
        prob=0.01,
        all_centroids=None,
        matched_centroids=None,
        image_size=None,
    )
    eng._imu_service = MagicMock()
    eng._imu_camera_from_telescope = None

    eng._capture_plate_solve_anchor(
        eng.pointing_state.read(), time.monotonic(), 1
    )

    assert eng.plate_solve_anchor is None
    eng._imu_service.orientation_at_host_time.assert_not_called()


def test_imu_worker_updates_pointing_from_fresh_sample(tmp_path):
    eng = _make_engine(tmp_path)
    _prepare_imu_worker_test(eng)

    eng._propagate_imu_pointing()

    result = eng.pointing_state.read()
    assert abs(result.ra_j2000 - 90.0) < 1e-7
    assert abs(result.dec_j2000) < 1e-7
    assert result.roll == 0.0


def test_imu_worker_ignores_stale_sample(tmp_path):
    eng = _make_engine(tmp_path)
    _prepare_imu_worker_test(eng, sample_age=1.0)
    before = eng.pointing_state.read()

    eng._propagate_imu_pointing()

    after = eng.pointing_state.read()
    assert after.ra_j2000 == before.ra_j2000
    assert after.dec_j2000 == before.dec_j2000


def test_imu_worker_cannot_overwrite_newer_plate_solve(tmp_path):
    eng = _make_engine(tmp_path)
    _prepare_imu_worker_test(eng)

    # Simulate a newer successful solve after the worker's anchor was captured.
    eng.pointing_state.update(
        ra_j2000=210.0,
        dec_j2000=-15.0,
        roll=8.0,
        matches=24,
        prob=0.02,
        all_centroids=None,
        matched_centroids=None,
        image_size=None,
    )
    before = eng.pointing_state.read()

    eng._propagate_imu_pointing()

    after = eng.pointing_state.read()
    assert after.ra_j2000 == before.ra_j2000
    assert after.dec_j2000 == before.dec_j2000
    assert after.last_success_timestamp == before.last_success_timestamp


def test_save_imu_alignment_rejects_manual_verified_transforms(tmp_path):
    eng = _make_engine(tmp_path)

    with pytest.raises(
        ValueError,
        match="Manual alignment matrices are disabled",
    ):
        eng.save_imu_alignment(np.eye(3), np.eye(3), True)

    path = eng._config.path.parent / "imu" / "sensor_alignment.json"
    assert not path.exists()
def test_save_imu_alignment_requires_physical_confirmation(tmp_path):
    eng = _make_engine(tmp_path)
    with pytest.raises(ValueError):
        eng.save_imu_alignment(np.eye(3), np.eye(3), False)


def test_save_imu_alignment_rejects_reflection_matrix(tmp_path):
    eng = _make_engine(tmp_path)
    reflection = np.diag([1.0, 1.0, -1.0])
    with pytest.raises(ValueError):
        eng.save_imu_alignment(reflection, np.eye(3), True)


def test_runtime_alignment_status_does_not_trust_saved_flags_alone(tmp_path):
    import json

    eng = _make_engine(tmp_path)
    imu_dir = eng._config.path.parent / "imu"
    imu_dir.mkdir(parents=True, exist_ok=True)
    (imu_dir / "sensor_alignment.json").write_text(
        json.dumps({
            "mapping_status": "VERIFIED",
            "camera_from_telescope_status": "VERIFIED",
            "sensor_to_telescope": np.eye(3).tolist(),
            "camera_from_telescope": np.eye(3).tolist(),
        }),
        encoding="utf-8",
    )

    status = eng.imu_alignment_status()
    assert status["sensor_mapping_status"] == "UNVERIFIED"
    assert status["camera_alignment_status"] == "VERIFIED"
    assert status["celestial_propagation_enabled"] is False



def test_synthetic_anchor_uses_exact_frame_timestamp(tmp_path):
    from unittest.mock import MagicMock
    import numpy as np

    eng = _make_engine(tmp_path)
    pointing = _anchor_test_pointing(valid=True)

    imu = MagicMock()
    imu.sensor_mapping_verified = True
    imu.reference_initialized = True
    imu.orientation_at_host_time.return_value = (
        1_000_000,
        [1.0, 0.0, 0.0, 0.0],
    )
    eng._imu_service = imu
    eng._imu_camera_from_telescope = np.eye(3)

    frame_timestamp = 1000.0
    frame_id = 1001
    eng._capture_plate_solve_anchor(
        pointing,
        frame_timestamp=frame_timestamp,
        frame_id=frame_id,
    )

    imu.orientation_at_host_time.assert_called_once_with(frame_timestamp)
    anchor = eng._plate_solve_anchor
    assert anchor is not None
    assert anchor.frame_timestamp == frame_timestamp
    assert anchor.frame_id == frame_id
    assert anchor.imu_device_time_us == 1_000_000


def test_synthetic_anchor_is_not_created_without_frame_time_orientation(tmp_path):
    from unittest.mock import MagicMock
    import numpy as np

    eng = _make_engine(tmp_path)
    pointing = _anchor_test_pointing(valid=True)

    imu = MagicMock()
    imu.sensor_mapping_verified = True
    imu.reference_initialized = True
    imu.orientation_at_host_time.return_value = None
    eng._imu_service = imu
    eng._imu_camera_from_telescope = np.eye(3)
    eng._plate_solve_anchor = None

    eng._capture_plate_solve_anchor(
        pointing,
        frame_timestamp=1000.0,
        frame_id=1001,
    )

    assert eng._plate_solve_anchor is None

# Synthetic IMU motion through the production engine propagation path.

def _synthetic_engine_anchor(ra=0.0, dec=0.0, quaternion=(1.0, 0.0, 0.0, 0.0)):
    from types import SimpleNamespace

    return SimpleNamespace(
        ra_j2000=ra,
        dec_j2000=dec,
        quaternion=quaternion,
        telescope_to_sky=_telescope_to_sky(ra, dec),
        pointing_timestamp=123.0,
        frame_id=9001,
    )


def _synthetic_axis_quaternion(axis, degrees):
    import math

    angle = math.radians(degrees) / 2.0
    sine = math.sin(angle)
    return (
        math.cos(angle),
        axis[0] * sine,
        axis[1] * sine,
        axis[2] * sine,
    )


def _synthetic_angular_separation(ra1, dec1, ra2, dec2):
    import math

    a1, d1, a2, d2 = map(
        math.radians, (ra1, dec1, ra2, dec2)
    )
    dot = (
        math.sin(d1) * math.sin(d2)
        + math.cos(d1) * math.cos(d2) * math.cos(a1 - a2)
    )
    return math.degrees(math.acos(max(-1.0, min(1.0, dot))))


import pytest


@pytest.mark.parametrize("angle", [1.0, 5.0, 10.0, 15.0, -1.0, -5.0, -10.0])
def test_synthetic_altitude_motion_propagates_expected_angle(tmp_path, angle):
    eng = _make_engine(tmp_path)
    anchor = _synthetic_engine_anchor()
    current = _synthetic_axis_quaternion((0.0, 1.0, 0.0), angle)

    result = eng._imu_propagate_radec(anchor, current)

    assert result is not None
    ra, dec = result
    separation = _synthetic_angular_separation(
        anchor.ra_j2000, anchor.dec_j2000, ra, dec
    )
    assert separation == pytest.approx(abs(angle), abs=1e-7)


@pytest.mark.parametrize("ra,dec", [(0.0, 0.0), (120.0, 30.0), (359.0, 45.0), (80.0, 89.0)])
def test_synthetic_motion_has_correct_great_circle_distance_at_sky_positions(
    tmp_path, ra, dec
):
    eng = _make_engine(tmp_path)
    anchor = _synthetic_engine_anchor(ra, dec)
    current = _synthetic_axis_quaternion((0.0, 1.0, 0.0), 3.5)

    result = eng._imu_propagate_radec(anchor, current)

    assert result is not None
    assert _synthetic_angular_separation(ra, dec, *result) == pytest.approx(
        3.5, abs=1e-6
    )


@pytest.mark.parametrize("angle", [5.0, 30.0, 90.0, 180.0, -45.0])
def test_synthetic_roll_about_boresight_does_not_change_ra_dec(tmp_path, angle):
    eng = _make_engine(tmp_path)
    anchor = _synthetic_engine_anchor(123.0, 42.0)
    current = _synthetic_axis_quaternion((1.0, 0.0, 0.0), angle)

    result = eng._imu_propagate_radec(anchor, current)

    assert result is not None
    assert result[0] == pytest.approx(123.0, abs=1e-7)
    assert result[1] == pytest.approx(42.0, abs=1e-7)


def test_synthetic_relative_rotation_works_with_nonidentity_anchor(tmp_path):
    eng = _make_engine(tmp_path)
    import math

    anchor_q = _synthetic_axis_quaternion((1.0, 0.0, 0.0), 25.0)
    delta_q = _synthetic_axis_quaternion((0.0, 1.0, 0.0), 7.0)

    # q_current = q_anchor * q_delta; engine must compute q_anchor^-1 * q_current.
    current_q = eng._imu_quaternion_multiply(anchor_q, delta_q)
    anchor = _synthetic_engine_anchor(40.0, 20.0, anchor_q)

    result = eng._imu_propagate_radec(anchor, current_q)

    assert result is not None
    assert _synthetic_angular_separation(40.0, 20.0, *result) == pytest.approx(
        7.0, abs=1e-6
    )


@pytest.mark.parametrize("angle", [2.0, 5.0, 12.0, -4.0])
def test_synthetic_fresh_worker_sample_updates_pointing_by_motion_angle(
    tmp_path, angle
):
    eng = _make_engine(tmp_path)
    _prepare_imu_worker_test(eng)

    eng._imu_service.latest_snapshot.quaternion = (
        _synthetic_axis_quaternion((0.0, 1.0, 0.0), angle)
    )

    eng._propagate_imu_pointing()

    result = eng.pointing_state.read()
    separation = _synthetic_angular_separation(
        0.0, 0.0, result.ra_j2000, result.dec_j2000
    )
    assert separation == pytest.approx(abs(angle), abs=1e-6)


def test_synthetic_worker_roll_motion_preserves_pointing_coordinates(tmp_path):
    eng = _make_engine(tmp_path)
    _prepare_imu_worker_test(eng)
    before = eng.pointing_state.read()

    eng._imu_service.latest_snapshot.quaternion = (
        _synthetic_axis_quaternion((1.0, 0.0, 0.0), 45.0)
    )

    eng._propagate_imu_pointing()

    after = eng.pointing_state.read()
    assert after.ra_j2000 == pytest.approx(before.ra_j2000, abs=1e-7)
    assert after.dec_j2000 == pytest.approx(before.dec_j2000, abs=1e-7)
