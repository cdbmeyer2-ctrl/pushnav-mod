"""Unwrap the Arduino MPU6050 32-bit micros() timestamp."""
from __future__ import annotations

UINT32 = 1 << 32
HALF_UINT32 = 1 << 31


class DeviceTimestampUnwrapper:
    """Convert wrapping uint32 microsecond timestamps into increasing values."""

    def __init__(self):
        self._last_raw: int | None = None
        self._epoch = 0

    def unwrap(self, timestamp_us: int) -> int:
        if not isinstance(timestamp_us, int):
            raise TypeError("timestamp_us must be an integer")

        if not 0 <= timestamp_us < UINT32:
            raise ValueError("timestamp_us must be an unsigned 32-bit value")

        if self._last_raw is not None:
            if timestamp_us < self._last_raw:
                backward = self._last_raw - timestamp_us

                if backward > HALF_UINT32:
                    self._epoch += UINT32
                else:
                    raise ValueError("Out-of-order device timestamp")

            elif timestamp_us - self._last_raw > HALF_UINT32:
                raise ValueError("Stale timestamp from previous clock epoch")

        self._last_raw = timestamp_us
        return self._epoch + timestamp_us

    def reset(self) -> None:
        self._last_raw = None
        self._epoch = 0
