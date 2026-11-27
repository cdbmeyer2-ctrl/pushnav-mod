"""Thread-safe serial ingestion and timestamp-preserving IMU sample history."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import threading
import time

from .clock import DeviceTimestampUnwrapper
from .protocol import PacketParser, RawSample


@dataclass(frozen=True)
class ImuSampleRecord:
    """Decoded IMU sample with device and Windows receipt timestamps."""
    sample: RawSample
    received_at_monotonic: float
    batch_id: int
    device_time_us: int


class ImuSampleHistory:
    """Bounded, thread-safe history of decoded IMU samples."""

    def __init__(self, capacity: int = 10000):
        if capacity < 1:
            raise ValueError("capacity must be at least 1")
        self._samples = deque(maxlen=capacity)
        self._lock = threading.Lock()

    def add_batch(self, records: list[ImuSampleRecord]) -> None:
        with self._lock:
            self._samples.extend(records)

    def snapshot(self) -> tuple[ImuSampleRecord, ...]:
        with self._lock:
            return tuple(self._samples)

    def latest(self) -> ImuSampleRecord | None:
        with self._lock:
            return self._samples[-1] if self._samples else None

    def samples_between(
        self, start_monotonic: float, end_monotonic: float
    ) -> tuple[ImuSampleRecord, ...]:
        if end_monotonic < start_monotonic:
            raise ValueError("end_monotonic precedes start_monotonic")
        with self._lock:
            return tuple(
                item for item in self._samples
                if start_monotonic <= item.received_at_monotonic <= end_monotonic
            )

    def device_samples_between(
        self, start_us: int, end_us: int
    ) -> tuple[ImuSampleRecord, ...]:
        if end_us < start_us:
            raise ValueError("end_us precedes start_us")
        with self._lock:
            return tuple(
                item for item in self._samples
                if start_us <= item.device_time_us <= end_us
            )

    def __len__(self) -> int:
        with self._lock:
            return len(self._samples)


class ImuSerialStream:
    """
    Background serial reader.

    device_time_us is the unwrapped Arduino measurement timeline.
    received_at_monotonic is the Windows packet-receipt timeline.
    These timestamps are not interchangeable.
    """

    def __init__(
        self,
        port: str = "COM3",
        baudrate: int = 115200,
        timeout: float = 0.05,
        history_capacity: int = 10000,
    ):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.history = ImuSampleHistory(history_capacity)

        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._reader = None
        self._error: str | None = None
        self._batch_id = 0
        self._rejected_timestamps = 0
        self._lock = threading.Lock()

    @property
    def error(self) -> str | None:
        with self._lock:
            return self._error

    @property
    def rejected_timestamps(self) -> int:
        with self._lock:
            return self._rejected_timestamps

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        if self.running:
            return

        self._stop_event.clear()

        with self._lock:
            self._error = None
            self._rejected_timestamps = 0

        self._thread = threading.Thread(
            target=self._run,
            name="PushNav-IMU",
            daemon=True,
        )
        self._thread.start()

    def _run(self) -> None:
        try:
            import serial
        except ImportError:
            with self._lock:
                self._error = "pyserial is not installed"
            return

        parser = PacketParser()
        device_clock = DeviceTimestampUnwrapper()

        try:
            self._reader = serial.Serial(
                self.port,
                baudrate=self.baudrate,
                timeout=self.timeout,
            )

            # The Uno resets when its serial port opens.
            time.sleep(2.0)
            self._reader.reset_input_buffer()

            while not self._stop_event.is_set():
                first = self._reader.read(1)
                if not first:
                    continue

                data = first
                waiting = self._reader.in_waiting
                if waiting:
                    data += self._reader.read(waiting)

                received_at = time.monotonic()
                samples = parser.feed(data)

                if not samples:
                    continue

                self._batch_id += 1
                records = []

                for sample in samples:
                    try:
                        unwrapped_time = device_clock.unwrap(
                            sample.timestamp_us
                        )
                    except (TypeError, ValueError):
                        with self._lock:
                            self._rejected_timestamps += 1
                        continue

                    records.append(
                        ImuSampleRecord(
                            sample=sample,
                            received_at_monotonic=received_at,
                            batch_id=self._batch_id,
                            device_time_us=unwrapped_time,
                        )
                    )

                if records:
                    self.history.add_batch(records)

        except Exception as exc:
            with self._lock:
                self._error = f"{type(exc).__name__}: {exc}"

        finally:
            if self._reader is not None:
                try:
                    self._reader.close()
                except Exception:
                    pass
                self._reader = None

    def stop(self, join_timeout: float = 2.0) -> None:
        self._stop_event.set()
        thread = self._thread

        if thread is not None:
            thread.join(timeout=join_timeout)

        if thread is not None and not thread.is_alive():
            self._thread = None
