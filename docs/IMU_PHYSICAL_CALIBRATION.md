# PushNav physical IMU alignment and validation

This procedure applies to the stock Apertura AD12 Dobsonian with the IMU
rigidly mounted to the optical tube. The tube does not rotate on this mount.
The telescope-body convention is +X toward the aperture, +Y transverse, +Z up
when the tube is horizontal. Raising the aperture is positive altitude;
counter-clockwise azimuth viewed from above is positive azimuth.

## Before starting

1. Secure the MPU-6050/Arduino assembly so it cannot move relative to the tube.
2. Connect the Arduino Uno on its configured serial port and confirm live,
   stable 100 Hz IMU samples.
3. Run stationary gyro calibration with the telescope completely still.
4. Keep the camera server on protocol v2. The native Windows DirectShow server
   advertises `protocol_version: 2`, `timestamp_clock: windows_qpc`, and sends
   `PNF2` frame metadata. Protocol v1/JPEG-only frames are not accepted for
   physical camera/IMU pairing.
5. Keep the telescope stationary whenever a camera calibration plate solve is
   being captured. QPC is sampled in the DirectShow SampleCB callback; it is
   not a guaranteed sensor exposure-midpoint timestamp.

## Sensor-to-telescope sweeps

The guided workflow requires four deliberate, continuous sweeps. Each sweep
must last 1–30 seconds, contain at least 50 usable samples, and integrate at
least 8 degrees of motion. Keep the telescope still at the start and end.

1. Start a positive altitude sweep, raise the aperture smoothly, then finish.
2. Start a negative altitude sweep, lower the aperture smoothly, then finish.
3. Start a positive azimuth sweep, rotate counter-clockwise viewed from above,
   then finish.
4. Start a negative azimuth sweep, rotate clockwise viewed from above, then
   finish.
5. Fit the sensor alignment. If direction, noise, or residual checks fail,
   repeat the affected sweeps rather than adjusting the matrix manually.

The workflow applies the fitted transform to the live estimator and clears its
old attitude history. Wait for the new gravity reference to initialize before
camera pairing.

## Camera-to-telescope pairs

1. Enable camera-pair capture after the sensor fit.
2. Move to a clearly different sky pointing, stop, and allow the tube to settle.
3. Wait for a fresh, valid plate solve while stationary. The workflow retains
   only one pair per pose; poses must differ by at least 5 degrees.
4. Repeat at four or more substantially different pointings, varying both
   altitude and azimuth. Avoid collecting multiple pairs from the same pose.
5. Stop camera-pair capture and fit camera alignment. The fit must pass its
   3-degree RMS limit and attitude-coverage checks.
6. Finalize only when both fits and the synchronized DirectShow QPC evidence
   are accepted.

## Post-calibration validation

1. Restart PushNav so the persisted sensor, camera, and timing statuses are
   loaded from the alignment file.
2. Confirm the status reports sensor mapping VERIFIED, camera alignment
   VERIFIED, and camera timing VERIFIED_CALLBACK_QPC.
3. With a fresh plate solve, move the telescope slowly through known altitude
   and azimuth changes. Compare propagated RA/Dec with subsequent plate solves.
4. Repeat at several sky positions and note error growth with time since solve.
   Because the MPU-6050 has no magnetometer, yaw/azimuth drift is expected.
   Re-solve periodically; do not assume indefinite absolute azimuth accuracy.
5. Interrupt the IMU stream deliberately. Propagation must stop, the continuity
   epoch must change, and a new plate solve must be required before propagation
   resumes.

## Timing limitation

The QPC timestamp is the time the DirectShow sample callback runs. The protocol
also carries the DirectShow sample start time, but does not yet map that stream
clock to QPC or establish the camera sensor's exposure midpoint. The QPC
handshake verifies the host clock domain only. Stationary camera-pair capture
makes geometric calibration insensitive to this callback latency; live
propagation should be accepted only after the post-calibration motion test
demonstrates adequate accuracy for the camera and exposure settings in use.
