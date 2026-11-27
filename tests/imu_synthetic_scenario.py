"""Generate deterministic synthetic plate-solve + IMU motion data.

Quaternion convention in this fixture: WXYZ.
Telescope boresight: +Z.
Altitude rotation axis: +Y.
Roll rotation axis: +Z (boresight).
All timestamps are synthetic host-clock seconds.
"""
import json
import math
from pathlib import Path


def quat_axis_angle(axis, angle_deg):
    """Return WXYZ quaternion for a rotation around a unit axis."""
    angle = math.radians(angle_deg)
    s = math.sin(angle / 2.0)
    return [
        math.cos(angle / 2.0),
        axis[0] * s,
        axis[1] * s,
        axis[2] * s,
    ]


def quat_multiply(a, b):
    """Hamilton product, WXYZ; composed rotation is a * b."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return [
        aw*bw - ax*bx - ay*by - az*bz,
        aw*bx + ax*bw + ay*bz - az*by,
        aw*by - ax*bz + ay*bw + az*bx,
        aw*bz + ax*by - ay*bx + az*bw,
    ]


def normalize(q):
    n = math.sqrt(sum(v*v for v in q))
    return [v / n for v in q]


def make_scenario():
    # Synthetic plate solve at t=1000.000.
    anchor = {
        "frame_id": 1001,
        "frame_timestamp": 1000.000,
        "ra_deg": 120.0,
        "dec_deg": 30.0,
        "roll_deg": 0.0,
        "valid": True,
        "imu_anchor_quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
        "sensor_mapping_status": "VERIFIED",
        "camera_alignment_status": "VERIFIED",
        "synthetic_only": True,
    }

    # Quaternion represents relative telescope orientation from anchor.
    # Angular rates are nominal controlled inputs for test/injection.
    motions = [
        # time, altitude degrees, roll degrees, gyro dps [x,y,z]
        (1000.000,  0.0,  0.0, [0.0,   0.0,  0.0]),
        (1000.250,  0.0,  0.0, [0.0,   0.0,  0.0]),
        (1000.500,  5.0,  0.0, [0.0,  20.0,  0.0]),
        (1000.750, 10.0,  0.0, [0.0,  20.0,  0.0]),
        (1001.000,  0.0,  0.0, [0.0, -40.0,  0.0]),
        (1001.250,  0.0, 15.0, [0.0,   0.0, 60.0]),
    ]

    samples = []
    for timestamp, altitude, roll, gyro in motions:
        q_alt = quat_axis_angle([0.0, 1.0, 0.0], altitude)
        q_roll = quat_axis_angle([0.0, 0.0, 1.0], roll)

        # Keep the order explicit: altitude then boresight roll.
        q = normalize(quat_multiply(q_alt, q_roll))

        samples.append({
            "host_timestamp": timestamp,
            "imu_device_time_us": int(timestamp * 1_000_000),
            "quaternion_wxyz": q,
            "gyro_dps_xyz": gyro,
            "altitude_command_deg": altitude,
            "roll_command_deg": roll,
        })

    return {
        "description": "Synthetic-only plate-solve anchor and controlled IMU sequence",
        "quaternion_convention": "WXYZ",
        "anchor": anchor,
        "samples": samples,
    }


if __name__ == "__main__":
    output = Path(__file__).parent / "data" / "imu_synthetic_anchor_motion.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(make_scenario(), indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote synthetic scenario: {output}")
