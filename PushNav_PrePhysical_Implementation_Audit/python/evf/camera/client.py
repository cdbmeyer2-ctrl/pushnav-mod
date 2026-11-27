# Copyright (C) 2026 Arun Venkataswamy
#
# This file is part of PushNav.
#
# PushNav is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# PushNav is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with PushNav. If not, see <https://www.gnu.org/licenses/>.

"""Camera TCP client â€” connects to the camera server subprocess.

Per SPEC_PROTOCOL_CAMERA.md Â§4â€“5 and SPEC_ARCHITECTURE.md Â§4.1/Â§7.
"""

import json
import logging
import socket
import struct
import threading
import time

from evf.camera.protocol import (
    MSG_CLOCK_SYNC,
    MSG_CLOCK_SYNC_REQUEST,
    decode_frame_payload,
    MSG_CONTROL_INFO,
    MSG_ERROR,
    MSG_FRAME,
    MSG_GET_CONTROLS,
    MSG_HELLO,
    MSG_SET_CONTROL,
    DisconnectError,
    encode_json_message,
    encode_message,
    read_message,
    type_name,
)
from evf.engine.frame_buffer import LatestFrame, FrameMetadata

logger = logging.getLogger(__name__)

PROTOCOL_VERSION = 2
_DEFAULT_HOST = "127.0.0.1"
_DEFAULT_PORT = 8764


class ProtocolError(Exception):
    """Raised on handshake or protocol version mismatch."""


class CameraClient:
    """TCP client that speaks the camera binary protocol.

    Connects, performs the HELLO handshake, receives CONTROL_INFO,
    then continuously receives FRAME messages into a LatestFrame buffer.
    """

    def __init__(
        self,
        frame_buffer: LatestFrame,
        host: str = _DEFAULT_HOST,
        port: int = _DEFAULT_PORT,
    ) -> None:
        self._frame_buffer = frame_buffer
        self._host = host
        self._port = port
        self._sock: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._frame_id = 0
        self._last_qpc_timestamp = 0.0
        self._clock_sync: dict = {}
        self._controls: list[dict] = []
        self._controls_lock = threading.Lock()
        self._hello_info: dict = {}
        self._on_controls_cb: callable | None = None
        self._on_error_cb: callable | None = None
        self._last_frame_time: float = 0.0
        self._connected = False

    # -- public API -----------------------------------------------------------

    def connect(self, timeout: float = 5.0) -> dict:
        """Connect and perform handshake. Returns the camera HELLO info dict.

        Raises:
            ProtocolError: On version mismatch or handshake failure.
            ConnectionRefusedError: If the server is not running.
            DisconnectError: If the connection drops during handshake.
        """
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.settimeout(timeout)
        self._sock.connect((self._host, self._port))
        logger.info("Connected to camera server at %s:%d", self._host, self._port)

        # Receive camera HELLO
        msg_type, payload = read_message(self._sock)
        if msg_type != MSG_HELLO:
            raise ProtocolError(f"Expected HELLO, got {type_name(msg_type)}")
        self._hello_info = json.loads(payload.decode("utf-8"))
        logger.info("Camera HELLO: %s", self._hello_info)

        # Validate protocol version
        cam_version = self._hello_info.get("protocol_version")
        if cam_version not in (1, 2):
            raise ProtocolError(f"Camera protocol version mismatch: unsupported version {cam_version}")

        our_hello = {"protocol_version": cam_version, "app": "evf", "app_version": "0.3.0"}
        self._sock.sendall(encode_json_message(MSG_HELLO, our_hello))

        msg_type, payload = read_message(self._sock)
        if msg_type != MSG_CONTROL_INFO:
            raise ProtocolError(f"Expected CONTROL_INFO, got {type_name(msg_type)}")
        self._update_controls(payload)

        if cam_version == 2:
            if self._hello_info.get("timestamp_clock") != "windows_qpc":
                raise ProtocolError("Protocol v2 requires windows_qpc timestamp clock")
            t0 = time.monotonic()
            self._sock.sendall(encode_message(MSG_CLOCK_SYNC_REQUEST))
            msg_type, payload = read_message(self._sock)
            t1 = time.monotonic()
            if msg_type != MSG_CLOCK_SYNC:
                raise ProtocolError(f"Expected CLOCK_SYNC, got {type_name(msg_type)}")
            try:
                sync = json.loads(payload.decode("utf-8"))
                qpc = int(sync["qpc_ticks"])
                frequency = int(sync["qpc_frequency"])
                if qpc <= 0 or frequency <= 0:
                    raise ValueError("non-positive QPC values")
                qpc_seconds = qpc / frequency
                if not (t0 - 5.0 <= qpc_seconds <= t1 + 5.0):
                    raise ValueError("QPC clock does not match host monotonic clock")
                self._clock_sync = {
                    "clock": "windows_qpc",
                    "frequency": frequency,
                    "round_trip_s": max(0.0, t1 - t0),
                    "verified_same_host_clock": True,
                }
            except (KeyError, TypeError, ValueError, UnicodeDecodeError,
                    json.JSONDecodeError) as exc:
                raise ProtocolError(f"Invalid camera clock sync: {exc}") from exc

        self._connected = True
        return self._hello_info

    def start_receiving(self) -> None:
        """Start the background frame-receiving thread."""
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._recv_loop, name="camera-recv", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        """Stop receiving and close the connection."""
        self._stop_event.set()
        self._connected = False
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None
        logger.info("Camera client stopped")

    def set_control(self, control_id: str, value: int) -> None:
        """Send a SET_CONTROL message."""
        if self._sock is None:
            return
        msg = encode_json_message(MSG_SET_CONTROL, {"id": control_id, "value": value})
        try:
            self._sock.sendall(msg)
        except OSError as exc:
            logger.error("Failed to send SET_CONTROL: %s", exc)

    def update_cached_control(self, control_id: str, value: int) -> None:
        """Optimistically update the cached current value for a control.

        Avoids UI displaying stale values before the server's CONTROL_INFO
        reply arrives.
        """
        with self._controls_lock:
            for ctrl in self._controls:
                if ctrl.get("id") == control_id:
                    ctrl["cur"] = value
                    break

    def get_controls(self) -> None:
        """Send a GET_CONTROLS request."""
        if self._sock is None:
            return
        try:
            self._sock.sendall(encode_message(MSG_GET_CONTROLS))
            logger.debug("Sent GET_CONTROLS")
        except OSError as exc:
            logger.error("Failed to send GET_CONTROLS: %s", exc)

    @property
    def controls(self) -> list[dict]:
        with self._controls_lock:
            return list(self._controls)

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def last_frame_time(self) -> float:
        return self._last_frame_time

    def on_controls_update(self, callback: callable) -> None:
        """Register a callback for CONTROL_INFO updates."""
        self._on_controls_cb = callback

    def on_error(self, callback: callable) -> None:
        """Register a callback for ERROR messages."""
        self._on_error_cb = callback

    # -- internal -------------------------------------------------------------

    def _recv_loop(self) -> None:
        assert self._sock is not None
        try:
            while not self._stop_event.is_set():
                try:
                    msg_type, payload = read_message(self._sock)
                except socket.timeout:
                    continue
                self._handle_message(msg_type, payload)
        except (DisconnectError, ConnectionResetError, BrokenPipeError, OSError) as exc:
            if not self._stop_event.is_set():
                logger.warning("Camera connection lost: %s", exc)
                self._connected = False

    def _handle_message(self, msg_type: int, payload: bytes) -> None:
        if msg_type == MSG_FRAME:
            receipt_time = time.monotonic()
            try:
                jpeg, metadata = decode_frame_payload(payload)
            except (ValueError, struct.error) as exc:
                logger.warning("Rejected malformed camera frame: %s", exc)
                return
            if not jpeg:
                logger.warning("Rejected empty camera frame")
                return
            frame_id = metadata["frame_id"]
            if frame_id is None:
                self._frame_id += 1
                frame_id = self._frame_id
                timestamp = receipt_time
            else:
                if frame_id <= self._frame_id:
                    logger.debug("Rejected duplicate/out-of-order frame %d", frame_id)
                    return
                timestamp = metadata["timestamp"]
                if timestamp <= self._last_qpc_timestamp:
                    logger.warning("Rejected non-monotonic QPC frame timestamp")
                    return
                if abs(timestamp - receipt_time) > 10.0:
                    logger.warning("Rejected QPC timestamp outside host clock domain")
                    return
                self._last_qpc_timestamp = timestamp
            self._frame_id = frame_id
            self._last_frame_time = receipt_time
            frame_meta = FrameMetadata(
                timestamp_source=metadata["timestamp_source"],
                timestamp_quality=metadata["timestamp_quality"],
                sample_time_100ns=metadata["sample_time_100ns"],
                sample_time_valid=metadata["sample_time_valid"],
                timing_uncertainty_s=metadata["timing_uncertainty_s"],
                clock_sync_verified=bool(self._clock_sync.get("verified_same_host_clock")),
            )
            self._frame_buffer.set(jpeg, timestamp, frame_id, frame_meta)
        elif msg_type == MSG_CONTROL_INFO:
            self._update_controls(payload)
        elif msg_type == MSG_ERROR:
            error_msg = payload.decode("utf-8", errors="replace")
            logger.error("Camera error: %s", error_msg)
            if self._on_error_cb:
                self._on_error_cb(error_msg)
        else:
            logger.debug("Ignoring unknown message type: %s", type_name(msg_type))

    def _update_controls(self, payload: bytes) -> None:
        try:
            data = json.loads(payload.decode("utf-8"))
            with self._controls_lock:
                self._controls = data.get("controls", [])
            logger.debug("CONTROL_INFO: %s", self._controls)
            if self._on_controls_cb:
                self._on_controls_cb(self._controls)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            logger.error("Bad CONTROL_INFO payload: %s", exc)

