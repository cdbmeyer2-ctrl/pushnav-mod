import math
import pytest

from evf.imu.estimator import RelativeAttitudeEstimator as Estimator
from evf.imu.interpolation import interpolate_quaternion


def test_slerp_midpoint_is_halfway_rotation():
    start = (1.0, 0.0, 0.0, 0.0)
    end = (math.cos(math.pi / 4), 0.0, 0.0, math.sin(math.pi / 4))
    actual = Estimator._q_slerp(start, end, 0.5)
    expected = (math.cos(math.pi / 8), 0.0, 0.0, math.sin(math.pi / 8))
    for a, b in zip(actual, expected):
        assert abs(a - b) < 1e-9


def test_slerp_identical_quaternions():
    q = (0.9238795, 0.0, 0.3826834, 0.0)
    actual = Estimator._q_slerp(q, q, 0.5)
    norm = math.sqrt(sum(x * x for x in actual))
    assert abs(norm - 1.0) < 1e-9
    assert all(abs(a - b) < 1e-9 for a, b in zip(actual, Estimator._q_normalize(q)))


def test_slerp_handles_equivalent_negative_quaternion():
    q = (math.cos(math.pi / 6), 0.0, 0.0, math.sin(math.pi / 6))
    actual = Estimator._q_slerp(q, tuple(-x for x in q), 0.5)
    assert abs(abs(sum(a * b for a, b in zip(actual, q))) - 1.0) < 1e-9


def test_slerp_rejects_fraction_outside_interval():
    q = (1.0, 0.0, 0.0, 0.0)
    with pytest.raises(ValueError):
        Estimator._q_slerp(q, q, 1.1)


def test_timestamp_interpolation_uses_bracketing_samples():
    start = (1.0, 0.0, 0.0, 0.0)
    end = (math.cos(math.pi / 4), 0.0, 0.0, math.sin(math.pi / 4))
    actual = interpolate_quaternion([(10000, start), (20000, end)], 15000)
    expected = (math.cos(math.pi / 8), 0.0, 0.0, math.sin(math.pi / 8))
    for a, b in zip(actual, expected):
        assert abs(a - b) < 1e-9


def test_timestamp_interpolation_does_not_extrapolate():
    q = (1.0, 0.0, 0.0, 0.0)
    samples = [(10000, q), (20000, q)]
    assert interpolate_quaternion(samples, 9000) is None
    assert interpolate_quaternion(samples, 21000) is None


def test_timestamp_interpolation_rejects_large_gap():
    q = (1.0, 0.0, 0.0, 0.0)
    assert interpolate_quaternion([(10000, q), (50000, q)], 30000) is None


def test_timestamp_interpolation_rejects_unordered_samples():
    q = (1.0, 0.0, 0.0, 0.0)
    with pytest.raises(ValueError):
        interpolate_quaternion([(20000, q), (10000, q)], 15000)
