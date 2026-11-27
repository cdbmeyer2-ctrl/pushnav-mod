from types import SimpleNamespace

from evf.imu.clock_sync import map_host_time_to_device_us


def record(batch, device_us, received):
    return SimpleNamespace(
        batch_id=batch,
        device_time_us=device_us,
        received_at_monotonic=received,
    )


def test_clock_mapping_uses_one_endpoint_per_batch():
    records = [
        record(1, 10000, 100.020),
        record(1, 20000, 100.020),
        record(2, 30000, 100.030),
        record(2, 40000, 100.040),
    ]

    assert map_host_time_to_device_us(records, 100.050) == 50000


def test_clock_mapping_uses_lowest_observed_receipt_offset():
    records = [
        record(1, 10000, 100.015),
        record(2, 20000, 100.020),
    ]

    assert map_host_time_to_device_us(records, 100.050) == 50000


def test_clock_mapping_returns_none_without_records():
    assert map_host_time_to_device_us([], 100.0) is None


def test_clock_mapping_limits_observation_window():
    records = [
        record(1, 10000, 100.100),
        record(2, 20000, 100.020),
    ]

    assert map_host_time_to_device_us(
        records, 100.050, max_batches=1
    ) == 50000
