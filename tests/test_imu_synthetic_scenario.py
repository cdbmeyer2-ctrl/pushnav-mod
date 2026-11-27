"""Regression tests for the synthetic IMU/plate-solve scenario."""
import math

from pathlib import Path
import importlib.util

_helper_path = Path(__file__).with_name("imu_synthetic_scenario.py")
_spec = importlib.util.spec_from_file_location(
    "imu_synthetic_scenario", _helper_path
)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Cannot load synthetic scenario helper: {_helper_path}")

_scenario = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_scenario)
make_scenario = _scenario.make_scenario
def qmul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return [
        aw*bw - ax*bx - ay*by - az*bz,
        aw*bx + ax*bw + ay*bz - az*by,
        aw*by - ax*bz + ay*bw + az*bx,
        aw*bz + ax*by - ay*bx + az*bw,
    ]


def qrotate(q, v):
    qc = [q[0], -q[1], -q[2], -q[3]]
    rotated = qmul(qmul(q, [0.0, *v]), qc)
    return rotated[1:]


def angular_separation_arcsec(ra1, dec1, ra2, dec2):
    """Great-circle angular separation; handles RA wrap and poles."""
    r1, d1, r2, d2 = map(math.radians, [ra1, dec1, ra2, dec2])
    a = (
        math.sin((d2-d1)/2)**2
        + math.cos(d1)*math.cos(d2)*math.sin((r2-r1)/2)**2
    )
    a = min(1.0, max(0.0, a))
    return math.degrees(2*math.asin(math.sqrt(a))) * 3600.0


def scenario():
    return make_scenario()


def test_synthetic_anchor_has_expected_metadata():
    data = scenario()
    anchor = data["anchor"]
    assert anchor["frame_id"] == 1001
    assert anchor["frame_timestamp"] == 1000.0
    assert anchor["ra_deg"] == 120.0
    assert anchor["dec_deg"] == 30.0
    assert anchor["valid"] is True
    assert anchor["synthetic_only"] is True


def test_synthetic_anchor_is_identity_quaternion():
    q = scenario()["anchor"]["imu_anchor_quaternion_wxyz"]
    assert q == [1.0, 0.0, 0.0, 0.0]


def test_scenario_has_six_ordered_samples():
    samples = scenario()["samples"]
    assert len(samples) == 6
    times = [s["host_timestamp"] for s in samples]
    assert times == sorted(times)
    assert len(set(times)) == len(times)


def test_device_timestamps_match_host_seconds_fixture():
    for sample in scenario()["samples"]:
        assert sample["imu_device_time_us"] == int(
            sample["host_timestamp"] * 1_000_000
        )


def test_all_synthetic_quaternions_are_unit_length():
    for sample in scenario()["samples"]:
        q = sample["quaternion_wxyz"]
        norm = math.sqrt(sum(x*x for x in q))
        assert abs(norm - 1.0) < 1e-12


def test_no_motion_sample_is_identity():
    q = scenario()["samples"][1]["quaternion_wxyz"]
    assert q == [1.0, 0.0, 0.0, 0.0]


def test_five_degree_altitude_rotation_moves_boresight():
    q = scenario()["samples"][2]["quaternion_wxyz"]
    v = qrotate(q, [0.0, 0.0, 1.0])
    assert abs(v[0] - math.sin(math.radians(5))) < 1e-12
    assert abs(v[1]) < 1e-12
    assert abs(v[2] - math.cos(math.radians(5))) < 1e-12


def test_ten_degree_altitude_rotation_moves_boresight():
    q = scenario()["samples"][3]["quaternion_wxyz"]
    v = qrotate(q, [0.0, 0.0, 1.0])
    assert abs(v[0] - math.sin(math.radians(10))) < 1e-12
    assert abs(v[2] - math.cos(math.radians(10))) < 1e-12


def test_return_to_anchor_sample_is_identity():
    q = scenario()["samples"][4]["quaternion_wxyz"]
    assert q == [1.0, 0.0, 0.0, 0.0]


def test_optical_axis_roll_preserves_boresight():
    q = scenario()["samples"][5]["quaternion_wxyz"]
    v = qrotate(q, [0.0, 0.0, 1.0])
    assert abs(v[0]) < 1e-12
    assert abs(v[1]) < 1e-12
    assert abs(v[2] - 1.0) < 1e-12


def test_roll_sample_has_expected_angle():
    q = scenario()["samples"][5]["quaternion_wxyz"]
    angle = 2 * math.atan2(q[3], q[0])
    assert abs(math.degrees(angle) - 15.0) < 1e-12


def test_angular_separation_zero_is_zero():
    assert angular_separation_arcsec(120, 30, 120, 30) < 1e-8


def test_angular_separation_handles_ra_wraparound():
    error = angular_separation_arcsec(359.999, 0, 0.001, 0)
    assert error < 30.0


def test_angular_separation_handles_declination():
    error = angular_separation_arcsec(120, 30, 120, 30.001)
    assert abs(error - 3.6) < 1e-5


def test_angular_separation_is_symmetric():
    a = angular_separation_arcsec(120, 30, 121, 31)
    b = angular_separation_arcsec(121, 31, 120, 30)
    assert abs(a-b) < 1e-9

