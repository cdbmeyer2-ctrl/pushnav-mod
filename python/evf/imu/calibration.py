"""Stationary gyro-bias calibration helpers."""
from __future__ import annotations
from dataclasses import dataclass
from statistics import mean, pstdev
from .protocol import RawSample

@dataclass(frozen=True)
class Calibration:
    gyro_bias_dps: tuple[float, float, float]
    accel_mean_g: tuple[float, float, float]
    gyro_noise_dps: tuple[float, float, float]
    sample_count: int

    def to_dict(self) -> dict:
        return {
            "gyro_bias_dps": list(self.gyro_bias_dps),
            "accel_mean_g": list(self.accel_mean_g),
            "gyro_noise_dps": list(self.gyro_noise_dps),
            "sample_count": self.sample_count,
        }


def calibrate_stationary(samples: list[RawSample], minimum_samples: int = 200) -> Calibration:
    if len(samples) < minimum_samples:
        raise ValueError(f"Need at least {minimum_samples} samples; got {len(samples)}")
    gyro = [s.gyro_dps for s in samples]
    accel = [s.accel_g for s in samples]
    bias = tuple(mean(v[i] for v in gyro) for i in range(3))
    noise = tuple(pstdev(v[i] for v in gyro) for i in range(3))
    accel_mean = tuple(mean(v[i] for v in accel) for i in range(3))
    # Reject obvious movement during calibration.
    if max(noise) > 0.8:
        raise ValueError("Calibration appears disturbed: gyro noise too high. Keep telescope still and retry.")
    return Calibration(bias, accel_mean, noise, len(samples))
