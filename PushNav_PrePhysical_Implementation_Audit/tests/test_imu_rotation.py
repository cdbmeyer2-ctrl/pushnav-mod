"""Mathematical tests for IMU quaternion frame transformations."""

import math

from evf.imu.estimator import RelativeAttitudeEstimator as Estimator


def conjugate(q):
    return (q[0], -q[1], -q[2], -q[3])


def axis_rotation(axis, degrees):
    half = math.radians(degrees) / 2.0
    result = [math.cos(half), 0.0, 0.0, 0.0]
    result[axis + 1] = math.sin(half)
    return result


def relative_rotation(current, reference):
    return Estimator._q_normalize(
        Estimator._q_multiply(current, conjugate(reference))
    )


def test_ninety_degree_rotation_about_each_axis():
    cases = [
        ((0, 1, 0), (0, 0, 1)),
        ((0, 0, 1), (1, 0, 0)),
        ((1, 0, 0), (0, 1, 0)),
    ]

    for axis, (source, expected) in enumerate(cases):
        q = axis_rotation(axis, 90)
        actual = Estimator._q_rotate(q, source)

        for value, target in zip(actual, expected):
            assert abs(value - target) < 1e-9


def test_relative_rotation_uses_reference_frame_order():
    reference = axis_rotation(2, 30)
    change = axis_rotation(1, 90)

    current = Estimator._q_multiply(change, reference)

    actual_change = relative_rotation(current, reference)

    for actual, expected in zip(actual_change, change):
        assert abs(actual - expected) < 1e-9


def test_relative_rotation_moves_anchored_pointing_vector():
    reference = axis_rotation(2, 30)
    change = axis_rotation(1, 90)

    current = Estimator._q_multiply(change, reference)
    actual_change = relative_rotation(current, reference)

    anchor = (1.0, 0.0, 0.0)

    actual_direction = Estimator._q_rotate(actual_change, anchor)
    expected_direction = (0.0, 0.0, -1.0)

    for actual, expected in zip(actual_direction, expected_direction):
        assert abs(actual - expected) < 1e-9

