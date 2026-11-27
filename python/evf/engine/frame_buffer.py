# Copyright (C) 2026 Arun Venkataswamy
# This file is part of PushNav; see LICENSE for terms.

"""Thread-safe single-frame buffer retaining camera timestamp provenance."""

from dataclasses import dataclass
import threading


@dataclass(frozen=True)
class FrameMetadata:
    timestamp_source: str = "legacy_tcp_receipt"
    timestamp_quality: str = "untrusted"
    sample_time_100ns: int | None = None
    sample_time_valid: bool = False
    timing_uncertainty_s: float | None = None


class LatestFrame:
    """Latest-frame buffer; legacy get() tuple remains backward compatible."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._jpeg_bytes: bytes | None = None
        self._timestamp: float = 0.0
        self._frame_id: int = 0
        self._metadata = FrameMetadata()

    def set(self, jpeg_bytes: bytes, timestamp: float, frame_id: int,
            metadata: FrameMetadata | None = None) -> None:
        with self._lock:
            if frame_id <= self._frame_id:
                return
            self._jpeg_bytes = jpeg_bytes
            self._timestamp = timestamp
            self._frame_id = frame_id
            self._metadata = metadata or FrameMetadata()

    def get(self) -> tuple[bytes | None, float, int]:
        with self._lock:
            return self._jpeg_bytes, self._timestamp, self._frame_id

    def get_with_metadata(self):
        with self._lock:
            return (self._jpeg_bytes, self._timestamp, self._frame_id,
                    self._metadata)

    def clear(self) -> None:
        with self._lock:
            self._jpeg_bytes = None
            self._timestamp = 0.0
            self._frame_id = 0
            self._metadata = FrameMetadata()
