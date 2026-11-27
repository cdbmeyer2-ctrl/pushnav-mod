# Copyright (C) 2026 Arun Venkataswamy
# This file is part of PushNav; see LICENSE for terms.

"""Camera wire protocol. Version 2 adds per-frame DirectShow metadata."""

import json
import logging
import struct

logger = logging.getLogger(__name__)
_HEADER_FMT = "<II"
_HEADER_SIZE = 8
MAX_MESSAGE_SIZE = 16 * 1024 * 1024

MSG_HELLO = 0x00
MSG_FRAME = 0x01
MSG_CONTROL_INFO = 0x02
MSG_ERROR = 0x03
MSG_CLOCK_SYNC = 0x04
MSG_SET_CONTROL = 0x11
MSG_GET_CONTROLS = 0x12
MSG_CLOCK_SYNC_REQUEST = 0x13

FRAME_META_MAGIC = b"PNF2"
FRAME_META_FMT = "<4sQQQQI"
FRAME_META_SIZE = struct.calcsize(FRAME_META_FMT)
FRAME_FLAG_SAMPLE_TIME_VALID = 1

_TYPE_NAMES = {
    MSG_HELLO: "HELLO", MSG_FRAME: "FRAME", MSG_CONTROL_INFO: "CONTROL_INFO",
    MSG_ERROR: "ERROR", MSG_CLOCK_SYNC: "CLOCK_SYNC",
    MSG_SET_CONTROL: "SET_CONTROL", MSG_GET_CONTROLS: "GET_CONTROLS",
    MSG_CLOCK_SYNC_REQUEST: "CLOCK_SYNC_REQUEST",
}


class DisconnectError(Exception):
    """Raised when the remote end closes the connection mid-message."""


def encode_message(msg_type: int, payload: bytes = b"") -> bytes:
    if len(payload) > MAX_MESSAGE_SIZE:
        raise ValueError("camera message exceeds maximum size")
    return struct.pack(_HEADER_FMT, msg_type, len(payload)) + payload


def encode_json_message(msg_type: int, obj: dict) -> bytes:
    return encode_message(msg_type, json.dumps(obj).encode("utf-8"))


def encode_frame_metadata(frame_id: int, qpc_ticks: int, qpc_frequency: int,
                          sample_time_100ns: int = 0, flags: int = 0) -> bytes:
    if frame_id <= 0 or qpc_ticks <= 0 or qpc_frequency <= 0:
        raise ValueError("frame ID and QPC values must be positive")
    return struct.pack(FRAME_META_FMT, FRAME_META_MAGIC, frame_id, qpc_ticks,
                       qpc_frequency, sample_time_100ns, flags)


def decode_frame_payload(payload: bytes) -> tuple[bytes, dict]:
    """Decode v2 PNF2 metadata; legacy JPEG frames are marked untrusted."""
    if len(payload) >= FRAME_META_SIZE and payload[:4] == FRAME_META_MAGIC:
        _, frame_id, qpc, frequency, sample_time, flags = struct.unpack(
            FRAME_META_FMT, payload[:FRAME_META_SIZE]
        )
        if frame_id <= 0 or qpc <= 0 or frequency <= 0:
            raise ValueError("invalid frame timestamp metadata")
        return payload[FRAME_META_SIZE:], {
            "frame_id": frame_id, "timestamp": qpc / frequency,
            "timestamp_source": "directshow_sample_callback_qpc",
            "timestamp_quality": "callback",
            "sample_time_100ns": sample_time,
            "sample_time_valid": bool(flags & FRAME_FLAG_SAMPLE_TIME_VALID),
            "timing_uncertainty_s": None,
        }
    return payload, {
        "frame_id": None, "timestamp": None,
        "timestamp_source": "legacy_tcp_receipt",
        "timestamp_quality": "untrusted",
        "sample_time_100ns": None, "sample_time_valid": False,
        "timing_uncertainty_s": None,
    }


def _recv_exact(sock, n: int) -> bytes:
    if n < 0 or n > MAX_MESSAGE_SIZE:
        raise ValueError("invalid camera message length")
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise DisconnectError("Connection closed")
        buf.extend(chunk)
    return bytes(buf)


def read_message(sock) -> tuple[int, bytes]:
    header = _recv_exact(sock, _HEADER_SIZE)
    msg_type, length = struct.unpack(_HEADER_FMT, header)
    if length > MAX_MESSAGE_SIZE:
        raise ValueError(f"camera message too large: {length}")
    return msg_type, _recv_exact(sock, length) if length else b""


def type_name(msg_type: int) -> str:
    return _TYPE_NAMES.get(msg_type, f"UNKNOWN(0x{msg_type:02x})")
