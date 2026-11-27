from evf.imu.estimator import RelativeAttitudeEstimator


def interpolate_quaternion(samples, target_device_time_us, max_gap_us=30000):
    """Interpolate (device_time_us, quaternion) samples without extrapolation."""
    if max_gap_us <= 0:
        raise ValueError("max_gap_us must be positive")

    previous = None

    for timestamp, quaternion in samples:
        if not isinstance(timestamp, int):
            raise TypeError("device timestamps must be integers")

        if previous is not None:
            previous_time, previous_quaternion = previous

            if timestamp <= previous_time:
                raise ValueError("device timestamps must be strictly increasing")

            if timestamp == target_device_time_us:
                return tuple(RelativeAttitudeEstimator._q_normalize(quaternion))

            if previous_time < target_device_time_us < timestamp:
                gap = timestamp - previous_time
                if gap > max_gap_us:
                    return None

                fraction = (target_device_time_us - previous_time) / gap
                return tuple(RelativeAttitudeEstimator._q_slerp(
                    previous_quaternion, quaternion, fraction
                ))

        elif timestamp == target_device_time_us:
            return tuple(RelativeAttitudeEstimator._q_normalize(quaternion))

        previous = (timestamp, quaternion)

    return None
