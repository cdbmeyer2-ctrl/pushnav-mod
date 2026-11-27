"""Approximate mapping between Arduino and Windows monotonic clocks."""
from math import isfinite


def map_host_time_to_device_us(
    records, host_monotonic: float, max_batches: int = 256
) -> int | None:
    """
    Map a Windows monotonic timestamp to approximate Arduino microseconds.

    Uses the lowest observed host-receipt offset from one endpoint per batch.
    This estimates receipt-time alignment, not camera exposure time.
    """
    if not isfinite(host_monotonic):
        raise ValueError("host_monotonic must be finite")
    if max_batches < 1:
        raise ValueError("max_batches must be positive")

    endpoints = {}
    for record in records:
        endpoints[record.batch_id] = record

    recent = list(endpoints.values())[-max_batches:]
    if not recent:
        return None

    offsets = [
        record.received_at_monotonic - record.device_time_us / 1_000_000.0
        for record in recent
    ]

    estimated_device_us = round(
        (host_monotonic - min(offsets)) * 1_000_000.0
    )
    return estimated_device_us if estimated_device_us >= 0 else None
