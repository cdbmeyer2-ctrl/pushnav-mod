"""Binary serial packet decoder for PushNav_IMU_Uno. No hardware dependencies."""
from __future__ import annotations
from dataclasses import dataclass
import struct

SYNC = b"\xAA\x55"
PACKET = struct.Struct("<2sHIhhhhhhhB")
PACKET_SIZE = PACKET.size  # 23 bytes


def crc8_atm(data: bytes) -> int:
    crc = 0
    for byte in data:
        crc ^= byte
        for _ in range(8):
            crc = ((crc << 1) ^ 0x07) & 0xFF if crc & 0x80 else (crc << 1) & 0xFF
    return crc


@dataclass(frozen=True)
class RawSample:
    sequence: int
    timestamp_us: int
    ax: int
    ay: int
    az: int
    gx: int
    gy: int
    gz: int
    temperature_raw: int

    @property
    def accel_g(self) -> tuple[float, float, float]:
        return (self.ax / 16384.0, self.ay / 16384.0, self.az / 16384.0)

    @property
    def gyro_dps(self) -> tuple[float, float, float]:
        return (self.gx / 131.0, self.gy / 131.0, self.gz / 131.0)

    @property
    def temperature_c(self) -> float:
        return self.temperature_raw / 340.0 + 36.53


def decode_packet(frame: bytes) -> RawSample:
    if len(frame) != PACKET_SIZE:
        raise ValueError(f"Expected {PACKET_SIZE} bytes, got {len(frame)}")
    sync, seq, timestamp, ax, ay, az, gx, gy, gz, temp, crc = PACKET.unpack(frame)
    if sync != SYNC:
        raise ValueError("Bad sync")
    if crc8_atm(frame[:-1]) != crc:
        raise ValueError("Bad CRC")
    return RawSample(seq, timestamp, ax, ay, az, gx, gy, gz, temp)


class PacketParser:
    """Resynchronizing stream parser; safe with partial reads and corrupted bytes."""
    def __init__(self):
        self.buffer = bytearray()
        self.bad_crc = 0
        self.discarded_bytes = 0

    def feed(self, data: bytes) -> list[RawSample]:
        self.buffer.extend(data)
        samples: list[RawSample] = []
        while True:
            pos = self.buffer.find(SYNC)
            if pos < 0:
                keep = 1 if self.buffer.endswith(SYNC[:1]) else 0
                discard = len(self.buffer) - keep
                if discard:
                    del self.buffer[:discard]
                    self.discarded_bytes += discard
                break
            if pos:
                del self.buffer[:pos]
                self.discarded_bytes += pos
            if len(self.buffer) < PACKET_SIZE:
                break
            candidate = bytes(self.buffer[:PACKET_SIZE])
            try:
                sample = decode_packet(candidate)
            except ValueError:
                del self.buffer[0]
                self.bad_crc += 1
                continue
            samples.append(sample)
            del self.buffer[:PACKET_SIZE]
        return samples
