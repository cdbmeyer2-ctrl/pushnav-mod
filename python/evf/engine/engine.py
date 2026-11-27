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

"""Engine coordinator Ã¢â‚¬â€ wires all components together.

Per SPEC_ARCHITECTURE.md and impl0.md Ã‚Â§Phase 8.
Central coordinator that owns the lifecycle of every subsystem.
"""

import io
import json
import logging
import os
import re
import threading
import tempfile
from dataclasses import dataclass
import time
from pathlib import Path

from evf.camera.subprocess_mgr import SubprocessManager
from evf.config.logging_setup import setup_logging
from evf.config.manager import ConfigManager
from evf.engine.audio import AudioAlert
from evf.engine.frame_buffer import LatestFrame
from evf.engine.goto_target import GotoTarget
from evf.engine.pointing import PointingSnapshot, PointingState
from evf.engine.sample_injector import SampleInjector
from evf.engine.state import EngineState, StateMachine
from evf.engine.status_led import StatusLedController, find_leds
from evf.solver.solver import PlateSolver
import numpy as np

from evf.solver.sync import (
    SyncCandidate,
    auto_select,
    build_sync_candidates,
    compute_body_frame_sync,
    orientation_from_radec_roll,
)
from evf.network import local_ip
from evf.paths import version_json
from evf.solver.thread import SolverThread
from evf.lx200.server import Lx200Server
from evf.stellarium.server import StellariumServer
from evf.webserver.server import WebServer
from evf.imu.service import ImuService
from evf.imu.alignment_workflow import ImuAlignmentWorkflow

logger = logging.getLogger(__name__)

_VERSION_PATH = version_json()


def _read_app_version() -> str:
    """Read app_version string from VERSION.json at engine init.

    Falls back to '0.0.0' if the file is missing or malformed so that the
    LX200 :GVN# reply always returns a well-formed 'PushNav <version>#'.
    """
    try:
        with open(_VERSION_PATH) as f:
            data = json.load(f)
        return str(data.get("app_version", "0.0.0"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return "0.0.0"


_V4L2_SYSFS = Path("/sys/class/video4linux")
# A USB camera's sysfs device path runs through a root hub (".../usb3/3-2/...");
# SoC video blocks (Pi ISP, HEVC decoder) sit under /platform/ with no /usbN/.
_USB_DEVICE_PATH = re.compile(r"/usb\d+/")


def _usb_video_device_present() -> bool:
    """True if any USB video device is attached (Linux sysfs).

    Lets the headless camera watchdog skip spawning camera_server Ã¢â‚¬â€ and
    logging its failure Ã¢â‚¬â€ every few seconds while nothing is plugged in.
    Returns True when sysfs is unavailable, so the caller just tries.
    """
    if not _V4L2_SYSFS.is_dir():
        return True
    for node in _V4L2_SYSFS.iterdir():
        try:
            if _USB_DEVICE_PATH.search(str((node / "device").resolve())):
                return True
        except OSError:
            continue
    return False


@dataclass(frozen=True)
class PlateSolveAnchor:
    """A successful celestial solution paired with one IMU orientation."""
    ra_j2000: float
    dec_j2000: float
    roll: float
    pointing_timestamp: float
    frame_timestamp: float
    frame_id: int
    quaternion: tuple[float, float, float, float]
    imu_device_time_us: int
    telescope_to_sky: tuple[tuple[float, float, float], ...]


class Engine:
    """Central coordinator that owns all components and manages their lifecycle.

    Startup creates shared data structures (LatestFrame, PointingState,
    StateMachine, ConfigManager) and launches subsystems (camera subprocess,
    Stellarium server, solver).  The UI reads from these structures and
    calls back into the engine for tracking toggle and camera control changes.
    """

    def __init__(
        self,
        *,
        dev_mode: bool = False,
        config: ConfigManager | None = None,
    ) -> None:
        # Shared data structures
        self._frame_buffer = LatestFrame()
        self._pointing_state = PointingState()
        self._state_machine = StateMachine()
        self._config = config if config is not None else ConfigManager()
        self._goto_target = GotoTarget()
        self._app_version = _read_app_version()
        self._dev_mode = dev_mode

        # Sample injector (continuous JPEG injection for dev/debug)
        self._sample_injector = SampleInjector(self._frame_buffer)

        # Components (created during startup)
        self._solver: PlateSolver | None = None
        self._solver_thread: SolverThread | None = None
        self._audio: AudioAlert | None = None
        self._subprocess_mgr: SubprocessManager | None = None
        # Serializes camera (re)starts: the UI "Retry camera" button, the
        # headless watchdog and shutdown must never race each other.
        self._camera_lock = threading.Lock()
        self._camera_watchdog: threading.Thread | None = None
        self._status_led: StatusLedController | None = None
        # Set once shutdown() begins; stops the watchdog and refuses retries.
        self._shutting_down = threading.Event()
        # "Waiting for camera" is logged once per absence, not every attempt.
        self._camera_wait_logged = False
        self._stellarium: StellariumServer | None = None
        self._lx200: Lx200Server | None = None
        self._webserver: WebServer | None = None
        self._imu_service: ImuService | None = None
        self._imu_alignment_workflow: ImuAlignmentWorkflow | None = None
        self._imu_camera_capture_active = False
        self._imu_camera_from_telescope: np.ndarray | None = None
        self._plate_solve_anchor_lock = threading.Lock()
        self._plate_solve_anchor: PlateSolveAnchor | None = None
        self._imu_tracking_stop = threading.Event()
        self._imu_tracking_thread: threading.Thread | None = None

        # Sync state (two-phase calibration)
        self._sync_lock = threading.Lock()
        self._sync_error: str | None = None
        self._sync_candidates: list[SyncCandidate] | None = None
        self._sync_selected_idx: int | None = None
        self._sync_camera_ra: float = 0.0
        self._sync_camera_dec: float = 0.0
        self._sync_camera_roll: float = 0.0

    # -- properties -----------------------------------------------------------

    @property
    def frame_buffer(self) -> LatestFrame:
        return self._frame_buffer

    @property
    def pointing_state(self) -> PointingState:
        return self._pointing_state

    @property
    def state_machine(self) -> StateMachine:
        return self._state_machine

    @property
    def config(self) -> ConfigManager:
        return self._config

    @property
    def app_version(self) -> str:
        return self._app_version

    @property
    def dev_mode(self) -> bool:
        return self._dev_mode

    @property
    def sample_injector(self) -> SampleInjector:
        return self._sample_injector

    @property
    def imu_attitude(self):
        """Latest read-only relative IMU attitude, if available."""
        if self._imu_service is None:
            return None
        return self._imu_service.latest_attitude

    @property
    def imu_snapshot(self):
        """Latest synchronized IMU snapshot, if available."""
        if self._imu_service is None:
            return None
        return self._imu_service.latest_snapshot

    @property
    def plate_solve_anchor(self) -> PlateSolveAnchor | None:
        """Latest valid plate-solve/IMU anchor, if one was captured."""
        with self._plate_solve_anchor_lock:
            return self._plate_solve_anchor

    @property
    def camera_connected(self) -> bool:
        """True if the camera subprocess is running and connected."""
        return self._subprocess_mgr is not None and self._subprocess_mgr.running

    @property
    def camera_controls(self) -> list[dict] | None:
        """Return the current camera controls, or None if not connected."""
        if self._subprocess_mgr and self._subprocess_mgr.client:
            return self._subprocess_mgr.client.controls
        return None

    @property
    def consecutive_failures(self) -> int:
        if self._solver_thread:
            return self._solver_thread.consecutive_failures
        return 0

    @property
    def audio_enabled(self) -> bool:
        return self._config.audio_enabled

    @audio_enabled.setter
    def audio_enabled(self, value: bool) -> None:
        self._config.audio_enabled = value
        if self._audio:
            self._audio.enabled = value

    @property
    def goto_target(self) -> GotoTarget:
        return self._goto_target

    def clear_goto_target(self) -> None:
        self._goto_target.clear()

    def set_goto_target(self, ra_deg: float, dec_deg: float) -> None:
        """Set the GOTO target. Used by the catalog 'Set as target' flow."""
        self._goto_target.set(float(ra_deg), float(dec_deg))

    def set_audio_enabled(self, enabled: bool) -> None:
        self.audio_enabled = enabled

    def set_lx200_epoch(self, epoch: str) -> None:
        """Set the LX200 reported coordinate epoch ('jnow' or 'j2000').

        'jnow' is the LX200 standard (SkySafari / INDI / ASCOM / real Meade);
        'j2000' is for Stellarium Mobile PLUS, which reads LX200 coords as J2000.
        Persists to config and applies to the running server immediately.
        """
        epoch = epoch.lower()
        if epoch not in ("jnow", "j2000"):
            raise ValueError(f"lx200_epoch must be 'jnow' or 'j2000', got {epoch!r}")
        self._config.lx200_epoch = epoch
        if self._lx200 is not None:
            self._lx200.set_report_j2000(epoch == "j2000")

    @property
    def location(self) -> dict:
        """Resolve the active observer location.

        Priority:
          1. Stellarium status 'location' field, when a Stellarium client is
             actively reporting one (source='stellarium').
          2. ConfigManager.location, when manually set (source='manual').
          3. None Ã¢â‚¬â€ neither available (source=None, lat/lon both None).
        """
        if self._stellarium is not None:
            status = self._stellarium.stellarium_status
            if status:
                loc = status.get("location") if isinstance(status, dict) else None
                if loc and loc.get("latitude") is not None and loc.get("longitude") is not None:
                    return {
                        "latitude": float(loc["latitude"]),
                        "longitude": float(loc["longitude"]),
                        "source": "stellarium",
                    }
        manual = self._config.location
        if manual is not None:
            return {
                "latitude": manual[0],
                "longitude": manual[1],
                "source": "manual",
            }
        return {"latitude": None, "longitude": None, "source": None}

    def set_location(self, latitude: float | None, longitude: float | None) -> None:
        """Persist a manual location. Pass (None, None) to clear."""
        if latitude is None or longitude is None:
            self._config.location = None
        else:
            self._config.location = (float(latitude), float(longitude))

    def inject_sample(self, name: str | None) -> None:
        """Start/stop continuous injection of a sample image (dev only).

        name: one of {"a","b","c","d","orion"} or None to stop.
        """
        from evf.paths import samples_dir
        from evf.engine.sample_injector import load_sample_jpeg

        if not self._dev_mode:
            logger.warning("inject_sample called outside dev_mode Ã¢â‚¬â€ ignoring")
            return
        if name is None:
            self._sample_injector.set_jpeg(None, None)
            self._frame_buffer.clear()
            logger.info("Sample injection stopped")
            return
        sd = samples_dir()
        if sd is None:
            logger.error("Samples directory not available in this build")
            return
        try:
            jpeg = load_sample_jpeg(sd, name)
            self._sample_injector.set_jpeg(jpeg, name)
            logger.info("Sample injection started: %s.png", name)
        except Exception as exc:
            logger.error("Failed to load sample %s: %s", name, exc)

    def inject_target(self, ra_deg: float, dec_deg: float) -> None:
        """Set the GOTO target manually (dev only)."""
        if not self._dev_mode:
            logger.warning("inject_target called outside dev_mode Ã¢â‚¬â€ ignoring")
            return
        self._goto_target.set(ra_deg, dec_deg)
        logger.info("Dev: injected GOTO target at RA %.4fÃ‚Â° Dec %.4fÃ‚Â°", ra_deg, dec_deg)

    def capture_frame(self) -> Path | None:
        """Save the latest frame to ~/Downloads as PNG. Returns the path or None."""
        from datetime import datetime
        from PIL import Image

        jpeg, _ts, _fid = self._frame_buffer.get()
        if jpeg is None:
            return None
        img = Image.open(io.BytesIO(jpeg)).convert("RGB")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dest = Path.home() / "Downloads" / f"evf_capture_{timestamp}.png"
        dest.parent.mkdir(parents=True, exist_ok=True)
        img.save(dest)
        logger.info("Frame captured: %s", dest)
        return dest

    def set_min_matches(self, value: int) -> None:
        self._config.min_matches = int(value)

    def set_max_prob(self, value: float) -> None:
        self._config.max_prob = float(value)

    @property
    def stellarium_status(self) -> dict | None:
        if self._stellarium:
            return self._stellarium.stellarium_status
        return None

    @property
    def stellarium_object(self) -> dict | None:
        if self._stellarium:
            return self._stellarium.stellarium_object
        return None

    @property
    def stellarium_address(self) -> str | None:
        """Address to paste into desktop Stellarium's Telescope Control plugin.

        On a Raspberry Pi appliance the server binds 0.0.0.0, so advertise the
        LAN IP for desktop Stellarium running on another machine. On Mac /
        Windows / x86 Linux dev boxes the server binds 127.0.0.1 (same-machine
        use), so advertise "localhost". Falls back to "localhost" if the
        server is bound to 0.0.0.0 but no LAN IP is available.
        """
        if self._stellarium is None:
            return None
        port = self._stellarium.port
        if self._stellarium.host == "0.0.0.0":
            ip = local_ip()
            if ip is not None:
                return f"{ip}:{port}"
        return f"localhost:{port}"

    @property
    def lx200_address(self) -> str | None:
        """Address to paste into SkySafari / Stellarium Mobile / INDI / ASCOM.

        LX200 server binds 0.0.0.0 so we advertise the LAN IP (same IP the
        mobile web URL uses) so mobile apps on the same network can connect.

        Returns None when the LX200 server isn't running OR no LAN is
        available Ã¢â‚¬â€ the UI distinguishes these via the lx200_running flag.
        """
        if self._lx200 is None:
            return None
        ip = local_ip()
        if ip is None:
            return None
        return f"{ip}:{self._lx200.port}"

    @property
    def lx200_running(self) -> bool:
        """True if the LX200 server bound its socket successfully."""
        return self._lx200 is not None

    @property
    def stellarium_has_client(self) -> bool:
        """True if the Stellarium server currently has any client connected.

        Stellarium holds a persistent TCP connection, so this is a simple
        boolean Ã¢â‚¬â€ the UI lights the Stellarium activity indicator while it
        is True.
        """
        return self._stellarium is not None and self._stellarium.client_count > 0

    # LX200 clients (notably SkySafari in polling mode) open a fresh TCP
    # connection per command, so "has client connected right now" is almost
    # never True. Instead we expose the time since the last connect or
    # received command and hold the indicator lit for _LX200_INDICATOR_HOLD
    # seconds. The user asked for a minimum hold of 100 ms.
    _LX200_INDICATOR_HOLD = 0.1  # seconds

    @property
    def lx200_active(self) -> bool:
        """True if the LX200 server saw a connect or a command within the
        last _LX200_INDICATOR_HOLD seconds."""
        if self._lx200 is None:
            return False
        last = self._lx200.last_activity_monotonic
        if last == 0.0:
            return False
        return (time.monotonic() - last) < self._LX200_INDICATOR_HOLD

    # -- startup (Ã‚Â§8.2) -------------------------------------------------------

    def startup_logging(self) -> None:
        """Initialize logging and log version info."""
        setup_logging(verbose=self._config.verbose, console=True)
        logger.info("EVF starting up")
        self._log_version()

    @staticmethod
    def _load_imu_camera_alignment(imu_dir: Path) -> np.ndarray | None:
        """Load a verified telescope-body to camera-body rotation.

        sensor_alignment.json must contain both:
          mapping_status: "VERIFIED"
          camera_from_telescope_status: "VERIFIED"
          camera_from_telescope: proper 3x3 rotation matrix

        The matrix maps telescope-body vectors into camera-body coordinates.
        """
        alignment_path = Path(imu_dir) / "sensor_alignment.json"
        try:
            with alignment_path.open("r", encoding="utf-8-sig") as stream:
                alignment = json.load(stream)

            if alignment.get("mapping_status") != "VERIFIED":
                raise ValueError("IMU sensor-to-telescope mapping is not VERIFIED")
            if alignment.get("camera_from_telescope_status") != "VERIFIED":
                raise ValueError("camera_from_telescope calibration is not VERIFIED")
            if alignment.get("camera_timing_status") != "VERIFIED_CALLBACK_QPC":
                raise ValueError("camera frame clock/timing calibration is not VERIFIED")

            matrix = np.asarray(alignment["camera_from_telescope"], dtype=float)
            if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
                raise ValueError("camera_from_telescope must be a finite 3x3 matrix")
            if not np.allclose(matrix.T @ matrix, np.eye(3), atol=1e-3):
                raise ValueError("camera_from_telescope rows/columns are not orthonormal")
            if not np.isclose(np.linalg.det(matrix), 1.0, atol=1e-3):
                raise ValueError("camera_from_telescope must have determinant +1")
            return matrix
        except FileNotFoundError:
            logger.warning(
                "IMU pointing propagation disabled: %s is missing",
                alignment_path,
            )
        except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError) as exc:
            logger.warning("IMU pointing propagation disabled: %s", exc)
        return None

    def startup_imu(self) -> None:
        """Start the optional background telescope IMU reader."""
        imu_dir = self._config.path.parent / "imu"
        self._imu_camera_from_telescope = self._load_imu_camera_alignment(imu_dir)
        self._imu_service = ImuService(imu_dir, port=self._config.imu_port)
        self._imu_service.start()
        self._imu_alignment_workflow = ImuAlignmentWorkflow(self._imu_service)
        self._imu_camera_capture_active = False
        self._imu_tracking_stop.clear()
        self._imu_tracking_thread = threading.Thread(
            target=self._propagate_imu_pointing,
            name="imu-pointing",
            daemon=True,
        )
        self._imu_tracking_thread.start()

    def startup_solver(self) -> None:
        """Load tetra3rs database (~1s)."""
        try:
            self._solver = PlateSolver()
        except Exception as exc:
            logger.error("Failed to load tetra3rs database: %s", exc)

    def startup_stellarium(self) -> None:
        """Start Stellarium TCP server."""
        try:
            self._stellarium = StellariumServer(
                self._pointing_state,
                goto_target=self._goto_target,
                on_client_connected=self._flash_status_led,
            )
            self._stellarium.start()
        except Exception as exc:
            logger.error("Failed to start Stellarium server: %s", exc)
            self._stellarium = None

    def startup_lx200(self) -> None:
        """Start LX200 TCP server.

        Binds 0.0.0.0:4030 so mobile apps (SkySafari, Stellarium Mobile) can
        reach it on the LAN. Always-on. The reported coordinate epoch follows
        the lx200.epoch config (JNow standard; J2000 for Stellarium Mobile PLUS).
        """
        try:
            self._lx200 = Lx200Server(
                self._pointing_state,
                goto_target=self._goto_target,
                app_version=self._app_version,
                report_j2000=(self._config.lx200_epoch == "j2000"),
                on_client_connected=self._flash_status_led,
            )
            self._lx200.start()
        except Exception as exc:
            logger.error("Failed to start LX200 server: %s", exc)
            self._lx200 = None

    def startup_webserver(self) -> None:
        """Start mobile web interface server."""
        try:
            self._webserver = WebServer(
                self._pointing_state,
                self._state_machine,
                self._goto_target,
                self._config,
                on_client_connected=self._flash_status_led,
                frame_buffer=self._frame_buffer,
                imu_attitude=lambda: self.imu_attitude,
                imu_snapshot=lambda: self.imu_snapshot,
                stellarium_object=lambda: self.stellarium_object,
                camera_controls=lambda: self.camera_controls,
                sync_state=lambda: {
                    "in_progress": self.sync_in_progress,
                    "candidates": [
                        {"idx": i, "name": f"Star #{i + 1}",
                         "ra_deg": c.ra, "dec_deg": c.dec, "magnitude": c.mag,
                         "pixel_x": c.x, "pixel_y": c.y}
                        for i, c in enumerate(self.sync_candidates or [])
                    ],
                    "selected_idx": self.sync_selected_idx,
                    "error": self.sync_error,
                },
                activity=lambda: {
                    "stellarium": {
                        "active": self.stellarium_has_client,
                        "address": self.stellarium_address,
                        "status": self.stellarium_status,
                        "object": self.stellarium_object,
                    },
                    "lx200": {
                        "active": self.lx200_active,
                        "address": self.lx200_address,
                        "epoch": self._config.lx200_epoch,
                    },
                    "webserver": {"url": self.web_url},
                    "audio_enabled": self.audio_enabled,
                },
                stellarium_location=lambda: (
                    (self.stellarium_status or {}).get("location")
                    if self.stellarium_status else None
                ),
                location=lambda: self.location,
                dev_mode=self._dev_mode,
                sample_active=lambda: self._sample_injector.active_name,
                actions=self,
            )
            self._webserver.start()
        except Exception as exc:
            logger.error("Failed to start web server: %s", exc)
            self._webserver = None

    @property
    def web_url(self) -> str | None:
        """LAN URL for the mobile web interface, or None if not running."""
        if self._webserver:
            return self._webserver.url
        return None

    def startup_camera(self) -> None:
        """Spawn camera subprocess, connect, handshake, restore settings.

        On first launch (config has no saved exposure/gain), each control is
        initialized to the midpoint of the range the camera reports Ã¢â‚¬â€ which
        differs by OS/camera backend, so we can't hardcode sensible defaults.

        With no USB video device attached (Linux), skips spawning
        camera_server Ã¢â‚¬â€ it would only fail and log its allowlist.
        """
        if not _usb_video_device_present():
            self._log_waiting_for_camera()
            return
        try:
            self._subprocess_mgr = SubprocessManager(
                self._frame_buffer, self._state_machine, self._config
            )
            hello = self._subprocess_mgr.start()
            logger.info("Camera connected: %s", hello)
            self._camera_wait_logged = False

            client = self._subprocess_mgr.client
            if client:
                for ctrl in client.controls:
                    cid = ctrl.get("id")
                    if cid not in ("exposure", "gain"):
                        continue
                    saved = self._config.exposure if cid == "exposure" else self._config.gain
                    if saved is None:
                        cmin = ctrl.get("min", 0)
                        cmax = ctrl.get("max", 0)
                        value = (cmin + cmax) // 2
                        logger.info(
                            "First-run %s = %d (midpoint of [%d, %d])",
                            cid, value, cmin, cmax,
                        )
                        if cid == "exposure":
                            self._config.exposure = value
                        else:
                            self._config.gain = value
                    else:
                        value = saved
                    client.set_control(cid, value)
                    client.update_cached_control(cid, value)
        except Exception as exc:
            logger.error("Failed to start camera: %s", exc)
            self._subprocess_mgr = None

    def retry_camera(self) -> bool:
        """Re-attempt camera startup on demand (e.g. user just plugged it in).

        Idempotent: if a camera is already connected, returns True without
        doing anything. Otherwise drops any stale SubprocessManager,
        re-runs startup_camera, and brings up the solver thread on success.
        Same code path as the initial boot, so the platform-specific risk
        surface (binary resolution, _kill_stale_server, Popen, handshake)
        is identical to what already runs once at every launch.
        Returns the post-attempt camera_connected state.
        """
        with self._camera_lock:
            if self._shutting_down.is_set():
                return False
            if self.camera_connected:
                return True
            # Stop the stale manager (ends any recovery loop still running)
            # and drop it so startup_camera builds a fresh one;
            # _kill_stale_server inside SubprocessManager._spawn_process will
            # reap any orphaned camera_server left over from the prior failure.
            stale, self._subprocess_mgr = self._subprocess_mgr, None
            if stale is not None:
                try:
                    stale.stop()
                except Exception as exc:
                    logger.warning("Error stopping stale camera manager: %s", exc)
            self.startup_camera()
            if not self.camera_connected:
                return False
            # Recovery may have left the engine in RECONNECTING / ERROR.
            if self._state_machine.state in (
                EngineState.RECONNECTING, EngineState.ERROR
            ):
                self._state_machine.transition(EngineState.SETUP)
            if self._solver_thread is None:
                self.startup_solver_thread()
            return True

    def start_camera_watchdog(self, interval_s: float = 10.0) -> None:
        """Keep retrying the camera in the background (headless mode).

        A headless Pi has no "Retry camera" click to fall back on: the
        service may boot before the camera is plugged in, and the
        SubprocessManager's own recovery gives up after 5 attempts. Every
        `interval_s` this checks for a missing camera and, once a USB
        video device is present, calls retry_camera(). It stays out of
        the way while SubprocessManager is mid-recovery.
        """
        if self._camera_watchdog is not None:
            return
        self._camera_watchdog = threading.Thread(
            target=self._camera_watchdog_loop,
            args=(interval_s,),
            name="camera-watchdog",
            daemon=True,
        )
        self._camera_watchdog.start()

    def start_status_led(self) -> None:
        """Show engine status on the Pi's LEDs (headless mode).

        Needs LEDs this process may write, which only the
        pushnav-headless service's root helper (evf.engine.led_helper)
        sets up; anywhere else this logs one line and does nothing.
        """
        if not self._config.led_enabled:
            logger.info("Status LED: disabled in config (led.enabled)")
            return
        leds = find_leds()
        if not leds:
            logger.info("Status LED: no writable LEDs, not used")
            return
        self._status_led = StatusLedController(
            lambda: (
                self._state_machine.state,
                self.camera_connected,
                self.consecutive_failures,
            ),
            leds,
        )
        self._status_led.start()
        self._goto_target.add_on_change(self._flash_on_new_target)
        logger.info("Status LED: using %s", ", ".join(sorted(leds)))

    def _flash_status_led(self) -> None:
        """Brief LED flash for a client connecting or sending a target.

        Called from the server threads; a no-op until start_status_led()
        has found writable LEDs.
        """
        led = self._status_led
        if led is not None:
            led.flash()

    def _flash_on_new_target(self) -> None:
        # GotoTarget also fires on clear(); only a new target flashes.
        if self._goto_target.read().active:
            self._flash_status_led()

    def _log_waiting_for_camera(self) -> None:
        if not self._camera_wait_logged:
            logger.info("Waiting for camera Ã¢â‚¬â€ plug it in to continue")
            self._camera_wait_logged = True

    def _camera_watchdog_loop(self, interval_s: float) -> None:
        while not self._shutting_down.wait(interval_s):
            mgr = self._subprocess_mgr
            if self.camera_connected or (mgr is not None and mgr.recovering):
                self._camera_wait_logged = False
                continue
            if not _usb_video_device_present():
                self._log_waiting_for_camera()
                continue
            if self.retry_camera():
                logger.info("Camera connected by watchdog")
                self._camera_wait_logged = False

    @staticmethod
    def _imu_quaternion_multiply(a, b):
        """Hamilton product for [w, x, y, z] quaternions."""
        aw, ax, ay, az = a
        bw, bx, by, bz = b
        return (
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        )

    @classmethod
    def _imu_rotate_vector(cls, quaternion, vector):
        """Rotate a 3-vector by a [w, x, y, z] unit quaternion."""
        w, x, y, z = quaternion
        vx, vy, vz = vector
        qv = (0.0, vx, vy, vz)
        qc = (w, -x, -y, -z)
        rotated = cls._imu_quaternion_multiply(
            cls._imu_quaternion_multiply(quaternion, qv), qc
        )
        return rotated[1], rotated[2], rotated[3]

    @classmethod
    def _imu_propagate_radec(cls, anchor, current_quaternion):
        """Propagate the telescope boresight through a calibrated sky transform.

        Estimator q maps telescope-body vectors into its fixed local frame.
        The anchor's telescope_to_sky matrix maps telescope-body vectors at
        the plate solve into celestial coordinates. Therefore the relative
        body rotation is inverse(q_anchor) * q_current, not a local-frame
        delta applied directly to a celestial vector.
        """
        q0 = tuple(anchor.quaternion)
        q1 = tuple(current_quaternion)
        if len(q0) != 4 or len(q1) != 4:
            return None

        norm0 = sum(value * value for value in q0) ** 0.5
        norm1 = sum(value * value for value in q1) ** 0.5
        if norm0 < 1e-9 or norm1 < 1e-9:
            return None
        q0 = tuple(value / norm0 for value in q0)
        q1 = tuple(value / norm1 for value in q1)

        telescope_to_sky = np.asarray(anchor.telescope_to_sky, dtype=float)
        if telescope_to_sky.shape != (3, 3) or not np.isfinite(telescope_to_sky).all():
            return None

        q0_inverse = (q0[0], -q0[1], -q0[2], -q0[3])
        relative_body = cls._imu_quaternion_multiply(q0_inverse, q1)
        relative_norm = sum(value * value for value in relative_body) ** 0.5
        if relative_norm < 1e-9:
            return None
        relative_body = tuple(value / relative_norm for value in relative_body)

        # Telescope frame: +X points toward the aperture; +Z is up.
        current_boresight_body = cls._imu_rotate_vector(
            relative_body, (1.0, 0.0, 0.0)
        )
        sky_vector = telescope_to_sky @ np.asarray(current_boresight_body)
        sky_norm = float(np.linalg.norm(sky_vector))
        if sky_norm < 1e-9 or not np.isfinite(sky_norm):
            return None
        x, y, z = sky_vector / sky_norm

        import math
        new_ra = math.degrees(math.atan2(y, x)) % 360.0
        new_dec = math.degrees(math.atan2(z, math.hypot(x, y)))
        return new_ra, new_dec

    def _propagate_imu_pointing(self) -> None:
        """Continuously propagate the last plate solve using relative IMU motion."""
        import math

        last_key = None
        while not self._imu_tracking_stop.wait(0.02):
            service = self._imu_service
            if service is None:
                continue

            anchor = self.plate_solve_anchor
            if anchor is None:
                continue

            try:
                snapshot = service.latest_snapshot
                if snapshot is None:
                    continue
                if time.monotonic() - snapshot.received_at_monotonic > 0.25:
                    with self._plate_solve_anchor_lock:
                        if self._plate_solve_anchor == anchor:
                            self._plate_solve_anchor = None
                    continue
                if getattr(snapshot, "continuity_id", 0) != anchor.continuity_id:
                    with self._plate_solve_anchor_lock:
                        if self._plate_solve_anchor == anchor:
                            self._plate_solve_anchor = None
                    logger.warning(
                        "IMU continuity epoch changed; waiting for a fresh plate solve"
                    )
                    continue

                key = (anchor.frame_id, snapshot.device_time_us)
                if key == last_key:
                    continue

                propagated = self._imu_propagate_radec(
                    anchor, snapshot.quaternion
                )
                if propagated is None:
                    continue

                ra, dec = propagated
                if not (math.isfinite(ra) and math.isfinite(dec)):
                    continue

                applied = self._pointing_state.update_from_imu(
                    ra,
                    dec,
                    expected_last_success_timestamp=anchor.pointing_timestamp,
                )
                if applied:
                    last_key = key
            except Exception:
                logger.exception("IMU pointing propagation failed")

    def _capture_plate_solve_anchor(
        self, pointing: PointingSnapshot, frame_timestamp: float, frame_id: int,
        frame_metadata=None,
    ) -> None:
        """Pair a plate solve with IMU orientation and a verified body transform."""
        workflow = self._imu_alignment_workflow
        if (
            workflow is not None
            and self._imu_camera_capture_active
            and pointing.valid
        ):
            try:
                workflow.capture_camera_pair(
                    pointing, frame_timestamp, frame_id, frame_metadata
                )
            except Exception as exc:
                logger.debug("IMU camera calibration pair rejected: %s", exc)

        service = self._imu_service
        camera_from_telescope = self._imu_camera_from_telescope

        sensor_mapping_verified = bool(
            service is not None
            and getattr(service, "sensor_mapping_verified", False)
        )

        timing_quality = getattr(frame_metadata, "timestamp_quality", "untrusted")
        timing_clock_verified = bool(
            getattr(frame_metadata, "clock_sync_verified", False)
        )
        if (
            not pointing.valid
            or service is None
            or not sensor_mapping_verified
            or camera_from_telescope is None
            or timing_quality != "callback"
            or not timing_clock_verified
        ):
            with self._plate_solve_anchor_lock:
                self._plate_solve_anchor = None
            logger.debug(
                "Plate-solve IMU anchor unavailable: verified sensor and "
                "camera/telescope alignment is required"
            )
            return

        orientation = service.orientation_at_host_time(frame_timestamp)
        if orientation is None:
            with self._plate_solve_anchor_lock:
                self._plate_solve_anchor = None
            logger.debug(
                "Plate-solve anchor unavailable: no initialized IMU orientation "
                "bracketing the camera-frame timestamp"
            )
            return

        device_time_us, quaternion = orientation
        current_snapshot = service.latest_snapshot
        if current_snapshot is None:
            with self._plate_solve_anchor_lock:
                self._plate_solve_anchor = None
            return
        camera_to_sky = orientation_from_radec_roll(
            pointing.ra_j2000, pointing.dec_j2000, pointing.roll
        )
        telescope_to_sky = camera_to_sky @ camera_from_telescope

        anchor = PlateSolveAnchor(
            ra_j2000=pointing.ra_j2000,
            dec_j2000=pointing.dec_j2000,
            roll=pointing.roll,
            pointing_timestamp=pointing.last_success_timestamp,
            frame_timestamp=frame_timestamp,
            frame_id=frame_id,
            quaternion=tuple(quaternion),
            imu_device_time_us=device_time_us,
            telescope_to_sky=tuple(
                tuple(float(value) for value in row)
                for row in telescope_to_sky
            ),
            continuity_id=getattr(current_snapshot, "continuity_id", 0),
        )
        with self._plate_solve_anchor_lock:
            self._plate_solve_anchor = anchor

    def startup_solver_thread(self) -> None:
        """Create solver thread object (not started until user enables tracking)."""
        if self._solver:
            self._audio = AudioAlert(enabled=self._config.audio_enabled)
            self._solver_thread = SolverThread(
                self._solver,
                self._frame_buffer,
                self._pointing_state,
                self._state_machine,
                self._config,
                audio=self._audio,
                on_plate_solve=self._capture_plate_solve_anchor,
            )
            # Wire solver failure count into web server for audio event detection
            if self._webserver is not None:
                self._webserver._solver_failures = lambda: self.consecutive_failures

    # -- tracking lifecycle (Ã‚Â§8.3) --------------------------------------------

    def skip_calibration(self) -> None:
        """Skip finder rotation calibration, proceed to tracking."""
        if self._state_machine.state == EngineState.CALIBRATE:
            if self._solver_thread:
                self._solver_thread.skip_calibrate()

    def use_previous_calibration(self) -> None:
        """Restore saved sync + finder rotation from config and start tracking.

        Only valid from SYNC state when config has saved calibration data.
        """
        if not self.camera_connected:
            logger.info("use_previous_calibration ignored Ã¢â‚¬â€ camera not connected")
            return
        if self._state_machine.state != EngineState.SYNC:
            return
        if not self._config.has_calibration:
            return
        if not self._solver_thread:
            logger.error("No solver thread available")
            return
        d_body = np.array(self._config.sync_d_body)
        self._solver_thread.set_sync_d_body(d_body)
        logger.info(
            "Using previous calibration: d_body=(%.6f, %.6f, %.6f) "
            "finder_rotation=%.1fÃ‚Â°",
            d_body[0], d_body[1], d_body[2],
            self._config.finder_rotation,
        )
        self._solver_thread.start()  # SYNC Ã¢â€ â€™ WARMING_UP

    def calibrate_imu_stationary(self) -> None:
        """Run stationary gyro calibration using the active IMU stream."""
        service = self._imu_service
        if service is None:
            raise RuntimeError("IMU is not running")
        service.calibrate_stationary_gyro()

    def imu_alignment_workflow_status(self) -> dict:
        workflow = self._imu_alignment_workflow
        if workflow is None:
            return {"available": False, "error": "IMU workflow is not running"}
        result = workflow.status()
        result["available"] = True
        result["camera_capture_active"] = self._imu_camera_capture_active
        return result

    def imu_alignment_start_sweep(self, axis: str, direction: int) -> dict:
        workflow = self._imu_alignment_workflow
        if workflow is None:
            raise RuntimeError("IMU workflow is not running")
        workflow.start_sensor_sweep(axis, direction)
        return self.imu_alignment_workflow_status()

    def imu_alignment_finish_sweep(self) -> dict:
        workflow = self._imu_alignment_workflow
        if workflow is None:
            raise RuntimeError("IMU workflow is not running")
        return workflow.finish_sensor_sweep()

    def imu_alignment_fit_sensor(self) -> dict:
        workflow = self._imu_alignment_workflow
        service = self._imu_service
        if workflow is None or service is None:
            raise RuntimeError("IMU workflow is not running")
        result = workflow.fit_sensor_alignment()
        service.set_sensor_to_telescope(result["matrix"])
        # Do not combine a newly fitted sensor transform with an old camera
        # transform. Propagation stays disabled until camera fit is finalized.
        self._imu_camera_from_telescope = None
        with self._plate_solve_anchor_lock:
            self._plate_solve_anchor = None
        return result

    def imu_alignment_camera_capture(self, active: bool) -> dict:
        workflow = self._imu_alignment_workflow
        if workflow is None:
            raise RuntimeError("IMU workflow is not running")
        if active and workflow.status()["sensor_fit"] is None:
            raise ValueError("Fit sensor axes before collecting camera pairs")
        self._imu_camera_capture_active = bool(active)
        return self.imu_alignment_workflow_status()

    def imu_alignment_fit_camera(self) -> dict:
        workflow = self._imu_alignment_workflow
        if workflow is None:
            raise RuntimeError("IMU workflow is not running")
        self._imu_camera_capture_active = False
        return workflow.fit_camera_alignment()

    def imu_alignment_reset(self) -> dict:
        workflow = self._imu_alignment_workflow
        if workflow is None:
            raise RuntimeError("IMU workflow is not running")
        self._imu_camera_capture_active = False
        workflow.reset()
        return self.imu_alignment_workflow_status()

    def finalize_imu_alignment(self) -> dict:
        workflow = self._imu_alignment_workflow
        service = self._imu_service
        if workflow is None or service is None:
            raise RuntimeError("IMU workflow is not running")
        measured = workflow.measured_result()
        sensor_matrix = np.asarray(measured["sensor_to_telescope"], dtype=float)
        camera_matrix = np.asarray(measured["camera_from_telescope"], dtype=float)
        if not getattr(service, "sensor_mapping_verified", False):
            raise RuntimeError("Sensor mapping is not applied and verified")

        imu_dir = self._config.path.parent / "imu"
        imu_dir.mkdir(parents=True, exist_ok=True)
        alignment_path = imu_dir / "sensor_alignment.json"
        existing = {}
        if alignment_path.exists():
            try:
                with alignment_path.open("r", encoding="utf-8-sig") as handle:
                    existing = json.load(handle)
                if not isinstance(existing, dict):
                    raise ValueError("Existing alignment must be a JSON object")
            except (OSError, ValueError) as exc:
                raise RuntimeError(f"Cannot preserve existing alignment: {exc}") from exc

        existing.update({
            "sensor_to_telescope": sensor_matrix.tolist(),
            "mapping_status": "VERIFIED",
            "camera_from_telescope": camera_matrix.tolist(),
            "camera_from_telescope_status": "VERIFIED",
            "camera_timing_status": "VERIFIED_CALLBACK_QPC",
            "camera_timing_note": (
                "QPC timestamp is sampled in DirectShow SampleCB. This verifies "
                "the host clock domain, not sensor exposure midpoint; perform "
                "physical stationary/motion validation before relying on tracking."
            ),
            "calibration_evidence": measured,
        })
        temp_name = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=imu_dir,
                prefix="sensor_alignment.", suffix=".tmp", delete=False,
            ) as handle:
                temp_name = handle.name
                json.dump(existing, handle, indent=2)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_name, alignment_path)
        except Exception:
            if temp_name and os.path.exists(temp_name):
                os.unlink(temp_name)
            raise

        self._imu_camera_from_telescope = camera_matrix
        self._imu_camera_capture_active = False
        with self._plate_solve_anchor_lock:
            self._plate_solve_anchor = None
        return {"saved": True, "evidence": measured}

    def imu_alignment_status(self) -> dict:
        """Return persisted transforms together with verified live runtime state."""
        alignment_path = self._config.path.parent / "imu" / "sensor_alignment.json"
        saved = {}
        try:
            with alignment_path.open("r", encoding="utf-8-sig") as handle:
                value = json.load(handle)
            if isinstance(value, dict):
                saved = value
        except (OSError, ValueError, TypeError):
            pass

        service = self._imu_service
        sensor_verified = bool(
            service is not None
            and getattr(service, "sensor_mapping_verified", False)
        )
        camera_verified = self._imu_camera_from_telescope is not None

        with self._plate_solve_anchor_lock:
            anchor_available = self._plate_solve_anchor is not None

        thread = self._imu_tracking_thread
        tracking_active = bool(thread is not None and thread.is_alive())

        snapshot_available = False
        if service is not None:
            try:
                snapshot_available = service.latest_snapshot is not None
            except Exception:
                snapshot_available = False

        return {
            "sensor_mapping_status": (
                "VERIFIED" if sensor_verified else "UNVERIFIED"
            ),
            "camera_alignment_status": (
                "VERIFIED" if camera_verified else "UNVERIFIED"
            ),
            "camera_timing_status": saved.get(
                "camera_timing_status", "UNVERIFIED"
            ),
            "sensor_to_telescope": (
                saved.get("sensor_to_telescope") if sensor_verified else None
            ),
            "camera_from_telescope": (
                self._imu_camera_from_telescope.tolist()
                if camera_verified else None
            ),
            "celestial_propagation_enabled": bool(
                sensor_verified
                and camera_verified
                and tracking_active
                and snapshot_available
                and anchor_available
            ),
        }

    def save_imu_alignment(
        self,
        sensor_to_telescope,
        camera_from_telescope,
        confirm_physical_alignment: bool,
    ) -> None:
        """Reject manual matrix submission; calibration must be evidence-based."""
        raise ValueError(
            "Manual alignment matrices are disabled. Use the guided physical "
            "sweep and paired plate-solve calibration workflow."
        )

    def step_advance(self) -> None:
        """Advance the wizard: SETUP Ã¢â€ â€™ SYNC Ã¢â€ â€™ SYNC_CONFIRM Ã¢â€ â€™ CALIBRATE Ã¢â€ â€™ WARMING_UP, or stop."""
        state = self._state_machine.state
        logger.info("step_advance called, state=%s", state.value)
        # Every forward step needs camera frames. Only the "stop tracking"
        # branch (WARMING_UP/TRACKING Ã¢â€ â€™ SETUP) works without a camera.
        if not self.camera_connected and state not in (
            EngineState.WARMING_UP, EngineState.TRACKING,
        ):
            logger.info("step_advance ignored Ã¢â‚¬â€ camera not connected")
            return
        if state == EngineState.SETUP:
            self._state_machine.transition(EngineState.SYNC)
        elif state == EngineState.SYNC:
            if not self._solver:
                logger.error("No solver available")
                return
            if not self._sync_lock.acquire(blocking=False):
                return  # solve already in progress
            self._sync_error = None
            threading.Thread(
                target=self._perform_sync_solve, name="sync-solve", daemon=True
            ).start()
        elif state == EngineState.SYNC_CONFIRM:
            self._confirm_sync()
        elif state == EngineState.CALIBRATE:
            self.skip_calibration()
        elif state in (EngineState.WARMING_UP, EngineState.TRACKING):
            if self._solver_thread:
                self._solver_thread.stop()
                self._solver_thread.set_sync_d_body(None)
            self._clear_sync_data()
            state = self._state_machine.state
            if state in (EngineState.WARMING_UP, EngineState.TRACKING):
                self._state_machine.transition(EngineState.SETUP)

    # -- two-phase sync -------------------------------------------------------

    def _perform_sync_solve(self) -> None:
        """Background thread: solve frame, build candidates, transition to SYNC_CONFIRM."""
        try:
            frame_data, _, _ = self._frame_buffer.get()
            if frame_data is None:
                self._sync_error = "No camera frame available"
                return
            result = self._solver.solve_frame(frame_data)
            if not PlateSolver.is_valid(
                result, self._config.min_matches, self._config.max_prob
            ):
                self._sync_error = (
                    "Plate solve failed \u2014 ensure stars are visible and retry"
                )
                return
            candidates = build_sync_candidates(
                result["matched_centroids"],
                result["matched_stars"],
                result["image_size"],
            )
            if not candidates:
                self._sync_error = (
                    "No suitable stars found \u2014 adjust framing and retry"
                )
                return
            self._sync_candidates = candidates
            self._sync_selected_idx = auto_select(candidates, result["image_size"])
            self._sync_camera_ra = result["RA"]
            self._sync_camera_dec = result["Dec"]
            self._sync_camera_roll = result["Roll"]
            self._state_machine.transition(EngineState.SYNC_CONFIRM)
        except Exception as exc:
            logger.error("Sync solve failed: %s", exc)
            self._sync_error = str(exc)
        finally:
            self._sync_lock.release()

    def _confirm_sync(self) -> None:
        """Compute body-frame sync from single point and start calibration."""
        if not self._sync_candidates or self._sync_selected_idx is None:
            return
        if not self._solver_thread:
            logger.error("No solver thread available")
            return
        target = self._sync_candidates[self._sync_selected_idx]
        d_body = compute_body_frame_sync(
            self._sync_camera_ra,
            self._sync_camera_dec,
            self._sync_camera_roll,
            target.ra,
            target.dec,
        )
        logger.info(
            "Body-frame sync: cam=(%.4f, %.4f, roll=%.1f) "
            "target=(%.4f, %.4f) star_mag=%.1f d_body=(%.6f, %.6f, %.6f)",
            self._sync_camera_ra, self._sync_camera_dec, self._sync_camera_roll,
            target.ra, target.dec, target.mag,
            d_body[0], d_body[1], d_body[2],
        )
        self._solver_thread.set_sync_d_body(d_body)
        self._config.sync_d_body = d_body.tolist()
        self._solver_thread.start_calibrate(
            self._sync_camera_ra,
            self._sync_camera_dec,
            self._sync_camera_roll,
        )  # SYNC_CONFIRM Ã¢â€ â€™ CALIBRATE

    def sync_retry(self) -> None:
        """Go back to SYNC to redo the current solve (preserves earlier pairs)."""
        if self._state_machine.state == EngineState.SYNC_CONFIRM:
            self._sync_candidates = None
            self._sync_selected_idx = None
            self._sync_error = None
            self._state_machine.transition(EngineState.SYNC)

    def set_sync_selected(self, idx: int) -> None:
        """Called by UI when user taps a candidate."""
        if self._sync_candidates and 0 <= idx < len(self._sync_candidates):
            self._sync_selected_idx = idx

    def _clear_sync_data(self) -> None:
        self._sync_candidates = None
        self._sync_selected_idx = None
        self._sync_error = None

    @property
    def sync_candidates(self) -> list[SyncCandidate] | None:
        return self._sync_candidates

    @property
    def sync_selected_idx(self) -> int | None:
        return self._sync_selected_idx

    @property
    def sync_error(self) -> str | None:
        return self._sync_error

    @property
    def sync_in_progress(self) -> bool:
        return self._sync_lock.locked()

    # -- camera control (Ã‚Â§8.4) ------------------------------------------------

    def set_control(self, control_id: str, value: int) -> None:
        """Send a camera control change and persist to config."""
        client = self._subprocess_mgr.client if self._subprocess_mgr else None
        if client:
            client.set_control(control_id, value)
        # Persist to config
        if control_id == "exposure":
            self._config.exposure = value
        elif control_id == "gain":
            self._config.gain = value

    # -- shutdown (Ã‚Â§8.5) ------------------------------------------------------

    def shutdown(self) -> None:
        """Graceful shutdown. Each step with independent timeout."""
        logger.info("Shutting down")
        self._shutting_down.set()

        # 0a. Stop the status LED first so it can't show a stale state while
        # subsystems go down (the service's ExecStopPost restores the LEDs).
        if self._status_led:
            try:
                self._status_led.stop()
            except Exception as exc:
                logger.error("Error stopping status LED: %s", exc)

        # 0. Stop sample injector (so it stops writing frames before solver stops)
        try:
            self._sample_injector.stop()
        except Exception as exc:
            logger.error("Error stopping sample injector: %s", exc)

        # 1. Stop solver thread
        if self._solver_thread:
            try:
                self._solver_thread.stop(timeout=2)
            except Exception as exc:
                logger.error("Error stopping solver: %s", exc)

        # 1b. Stop IMU pointing propagation before its source service.
        self._imu_tracking_stop.set()
        if self._imu_tracking_thread:
            self._imu_tracking_thread.join(timeout=1.0)
            self._imu_tracking_thread = None

        # 1c. Stop IMU reader
        if self._imu_service:
            try:
                self._imu_service.stop()
            except Exception as exc:
                logger.error("Error stopping IMU: %s", exc)

        # 2. Stop Stellarium server
        if self._stellarium:
            try:
                self._stellarium.stop(timeout=2)
            except Exception as exc:
                logger.error("Error stopping Stellarium: %s", exc)

        # 2b. Stop LX200 server
        if self._lx200:
            try:
                self._lx200.stop(timeout=2)
            except Exception as exc:
                logger.error("Error stopping LX200 server: %s", exc)

        # 3a. Stop web server
        if self._webserver:
            try:
                self._webserver.stop(timeout=2)
            except Exception as exc:
                logger.error("Error stopping web server: %s", exc)

        # 3. Terminate camera subprocess. Taking _camera_lock waits out any
        # retry already in flight (_shutting_down refuses new ones), so no
        # camera_server can be spawned after we've stopped the manager.
        locked = self._camera_lock.acquire(timeout=30)
        try:
            if self._subprocess_mgr:
                try:
                    self._subprocess_mgr.stop()
                except Exception as exc:
                    logger.error("Error stopping camera: %s", exc)
        finally:
            if locked:
                self._camera_lock.release()

        # 4. Save config
        try:
            self._config.save()
        except Exception as exc:
            logger.error("Error saving config: %s", exc)

        # 5. Flush logs
        logger.info("Shutdown complete")
        logging.shutdown()

    # -- internal -------------------------------------------------------------

    def _log_version(self) -> None:
        try:
            with open(_VERSION_PATH) as f:
                version = json.load(f)
            logger.info("EVF version: %s", version)
        except (FileNotFoundError, json.JSONDecodeError) as exc:
            logger.warning("Could not read VERSION.json: %s", exc)

