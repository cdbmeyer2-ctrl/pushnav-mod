"""Regression tests for IMU estimator timestamp handling."""

import pytest

from evf.imu.calibration import Calibration
from evf.imu.estimator import RelativeAttitudeEstimator
from evf.imu.protocol import RawSample


@pytest.fixture
def calibration():
    return Calibration(
        gyro_bias_dps=(0.0, 0.0, 0.0),
        accel_mean_g=(0.0, 0.0, 1.0),
        gyro_noise_dps=(0.0, 0.0, 0.0),
        sample_count=300,
    )


def make_sample(timestamp):
    return RawSample(
        1, timestamp,
        0, 0, 16384,
        0, 0, 131,
        0,
    )


def make_estimator(calibration):
    return RelativeAttitudeEstimator(calibration)


def test_unwrapped_timestamp_integration(calibration):
    estimator = make_estimator(calibration)
    estimator.update(make_sample(1_000_000), device_time_us=1_000_000)
    initial_q = tuple(estimator.q)

    estimator.update(make_sample(1_010_000), device_time_us=1_010_000)

    assert tuple(estimator.q) != initial_q


def test_duplicate_timestamp_is_rejected(calibration):
    estimator = make_estimator(calibration)
    estimator.update(make_sample(1_000_000), device_time_us=1_000_000)
    estimator.update(make_sample(1_010_000), device_time_us=1_010_000)
    previous_q = tuple(estimator.q)

    estimator.update(make_sample(1_010_000), device_time_us=1_010_000)

    assert tuple(estimator.q) == previous_q


def test_backward_timestamp_is_rejected(calibration):
    estimator = make_estimator(calibration)
    estimator.update(make_sample(1_000_000), device_time_us=1_000_000)
    estimator.update(make_sample(1_010_000), device_time_us=1_010_000)
    previous_q = tuple(estimator.q)

    estimator.update(make_sample(1_005_000), device_time_us=1_005_000)

    assert tuple(estimator.q) == previous_q
    assert estimator._last_timestamp == 1_010_000


def test_long_gap_reanchors_and_recovers(calibration):
    estimator = make_estimator(calibration)
    estimator.update(make_sample(1_000_000), device_time_us=1_000_000)
    estimator.update(make_sample(1_010_000), device_time_us=1_010_000)
    previous_q = tuple(estimator.q)

    estimator.update(make_sample(1_400_000), device_time_us=1_400_000)

    assert tuple(estimator.q) == previous_q
    assert estimator._last_timestamp == 1_400_000

    estimator.update(make_sample(1_410_000), device_time_us=1_410_000)

    assert tuple(estimator.q) != previous_q


def test_legacy_uint32_rollover(calibration):
    estimator = make_estimator(calibration)
    estimator.update(make_sample(0xFFFFFFF0))
    initial_q = tuple(estimator.q)

    estimator.update(make_sample(0x00002700))

    assert tuple(estimator.q) != initial_q

def test_initial_relative_attitude_is_zero_when_starting_tilted(calibration):
    estimator = make_estimator(calibration)

    tilted_sample = RawSample(
        1, 1_000_000,
        0, 16384, 0,
        0, 0, 131,
        0,
    )

    attitude = estimator.update(
        tilted_sample,
        device_time_us=1_000_000,
    )

    assert attitude.reference_initialized
    assert attitude.roll_deg == pytest.approx(0.0, abs=1e-6)
    assert attitude.pitch_deg == pytest.approx(0.0, abs=1e-6)
    assert attitude.yaw_relative_deg == pytest.approx(0.0, abs=1e-6)
    assert attitude.altitude_relative_deg == pytest.approx(0.0, abs=1e-6)
    assert attitude.azimuth_relative_deg == pytest.approx(0.0, abs=1e-6)
