import { useEffect, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { api } from "@/lib/api";
import type { EnginePayload } from "@/lib/types";

interface Props {
  state: EnginePayload;
}

function degrees(value: number | undefined): string {
  if (typeof value !== "number" || !Number.isFinite(value)) return "?";
  return `${value >= 0 ? "+" : ""}${value.toFixed(2)} deg`;
}

function axes(value: [number, number, number] | null | undefined, unit: string): string {
  if (!value || value.some((v) => !Number.isFinite(v))) return "?";
  return value.map((v) => v.toFixed(3)).join(" / ") + ` ${unit}`;
}

function Reading({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-center justify-between gap-3">
      <span className="text-sm text-muted-foreground">{label}</span>
      <span className="text-sm font-mono tabular-nums">{value}</span>
    </div>
  );
}

export function ImuStatus({ state }: Props) {
  const imu = state.imu?.available ? state.imu : null;
  const alignment = state.imu_alignment;
  const [calibrating, setCalibrating] = useState(false);
  const [calibrationMessage, setCalibrationMessage] = useState("");
  const [calibrationError, setCalibrationError] = useState(false);
  const [workflowBusy, setWorkflowBusy] = useState(false);
  const [workflowMessage, setWorkflowMessage] = useState("");
  const [workflowError, setWorkflowError] = useState(false);
  const [workflowStatus, setWorkflowStatus] = useState<Record<string, unknown>>({});
  useEffect(() => {
    let mounted = true;
    api.imuAlignmentWorkflowStatus().then((status) => {
      if (mounted) setWorkflowStatus(status);
    }).catch((error: unknown) => {
      if (!mounted) return;
      setWorkflowError(true);
      setWorkflowMessage(error instanceof Error ? error.message : "Could not read calibration workflow.");
    });
    return () => { mounted = false; };
  }, []);

  async function calibrateStationaryGyro() {
    if (!imu || calibrating) return;
    setCalibrating(true);
    setCalibrationMessage("");
    setCalibrationError(false);
    try {
      await api.calibrateImuStationary();
      setCalibrationMessage("Gyroscope calibration saved successfully.");
    } catch (error) {
      setCalibrationError(true);
      setCalibrationMessage(
        error instanceof Error ? error.message : "Gyroscope calibration failed.",
      );
    } finally {
      setCalibrating(false);
    }
  }

  async function refreshWorkflow() {
    try {
      setWorkflowStatus(await api.imuAlignmentWorkflowStatus());
    } catch (error) {
      setWorkflowError(true);
      setWorkflowMessage(error instanceof Error ? error.message : "Could not read calibration workflow.");
    }
  }

  async function runWorkflow(action: () => Promise<void>, success: string) {
    if (workflowBusy) return;
    setWorkflowBusy(true);
    setWorkflowMessage("");
    setWorkflowError(false);
    try {
      await action();
      await refreshWorkflow();
      setWorkflowMessage(success);
    } catch (error) {
      setWorkflowError(true);
      setWorkflowMessage(error instanceof Error ? error.message : "Calibration action failed.");
    } finally {
      setWorkflowBusy(false);
    }
  }

  const sensorVerified = alignment?.sensor_mapping_status === "VERIFIED";
  const cameraVerified = alignment?.camera_alignment_status === "VERIFIED";
  const propagationReady =
    Boolean(alignment?.celestial_propagation_enabled) &&
    sensorVerified &&
    cameraVerified;

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-primary flex items-center justify-between">
          IMU Status
          <span className={`text-xs font-normal ${imu ? "text-green-500" : "text-muted-foreground"}`}>
            {imu ? "Receiving data" : "Unavailable"}
          </span>
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        <div className="space-y-2 rounded-md border p-3">
          <div className="text-sm font-medium">Gyroscope calibration</div>
          <p className="text-xs text-muted-foreground">
            Place the telescope on a stable surface and keep it completely
            stationary during calibration. Recent IMU samples measure and save
            the gyroscope bias.
          </p>
          <button
            type="button"
            onClick={calibrateStationaryGyro}
            disabled={!imu || calibrating}
            className="w-full rounded-md bg-primary px-3 py-2 text-sm font-medium text-primary-foreground disabled:cursor-not-allowed disabled:opacity-50"
          >
            {calibrating ? "Calibrating..." : "Calibrate gyroscope"}
          </button>
          {calibrationMessage && (
            <p role="status" aria-live="polite" className={`break-words text-xs ${
              calibrationError ? "text-destructive" : "text-green-600"
            }`}>
              {calibrationMessage}
            </p>
          )}
        </div>

        <div className="space-y-2 rounded-md border p-3">
          <div className="text-sm font-medium">Evidence-based physical alignment</div>
          <Reading label="Altitude sweeps" value={String(workflowStatus.sensor_sweep_count ?? 0)} />
          <Reading label="Camera/IMU pairs" value={String(workflowStatus.camera_pair_count ?? 0)} />
          <Reading label="Pair capture" value={workflowStatus.camera_capture_active ? "Active" : "Stopped"} />
          <p className="text-xs text-muted-foreground">
            Perform deliberate sweeps in both directions for altitude and azimuth.
            Fit sensor axes first. Then enable camera-pair capture, move to several
            different telescope positions, and wait for successful plate solves.
          </p>
          <div className="grid grid-cols-2 gap-2">
            <button type="button" disabled={workflowBusy || !imu} onClick={() => runWorkflow(() => api.imuAlignmentStartSweep("altitude", 1), "Positive altitude sweep started")} className="rounded-md border px-2 py-2 text-xs disabled:opacity-50">Start + altitude</button>
            <button type="button" disabled={workflowBusy || !imu} onClick={() => runWorkflow(() => api.imuAlignmentStartSweep("altitude", -1), "Negative altitude sweep started")} className="rounded-md border px-2 py-2 text-xs disabled:opacity-50">Start − altitude</button>
            <button type="button" disabled={workflowBusy || !imu} onClick={() => runWorkflow(() => api.imuAlignmentStartSweep("azimuth", 1), "Positive azimuth sweep started")} className="rounded-md border px-2 py-2 text-xs disabled:opacity-50">Start + azimuth</button>
            <button type="button" disabled={workflowBusy || !imu} onClick={() => runWorkflow(() => api.imuAlignmentStartSweep("azimuth", -1), "Negative azimuth sweep started")} className="rounded-md border px-2 py-2 text-xs disabled:opacity-50">Start − azimuth</button>
          </div>
          <button type="button" disabled={workflowBusy} onClick={() => runWorkflow(api.imuAlignmentFinishSweep, "Sweep recorded")} className="w-full rounded-md bg-secondary px-3 py-2 text-sm disabled:opacity-50">Finish current sweep</button>
          <button type="button" disabled={workflowBusy} onClick={() => runWorkflow(api.imuAlignmentFitSensor, "Sensor fit applied")} className="w-full rounded-md bg-secondary px-3 py-2 text-sm disabled:opacity-50">Fit sensor axes</button>
          <button type="button" disabled={workflowBusy} onClick={() => runWorkflow(() => api.imuAlignmentCameraCapture(true), "Camera pair capture enabled")} className="w-full rounded-md bg-secondary px-3 py-2 text-sm disabled:opacity-50">Start camera pair capture</button>
          <button type="button" disabled={workflowBusy} onClick={() => runWorkflow(() => api.imuAlignmentCameraCapture(false), "Camera pair capture stopped")} className="w-full rounded-md bg-secondary px-3 py-2 text-sm disabled:opacity-50">Stop camera pair capture</button>
          <button type="button" disabled={workflowBusy} onClick={() => runWorkflow(api.imuAlignmentFitCamera, "Camera fit completed")} className="w-full rounded-md bg-secondary px-3 py-2 text-sm disabled:opacity-50">Fit camera alignment</button>
          <button type="button" disabled={workflowBusy} onClick={() => runWorkflow(api.imuAlignmentFinalize, "Evidence-based alignment saved")} className="w-full rounded-md bg-primary px-3 py-2 text-sm font-medium text-primary-foreground disabled:opacity-50">Finalize verified alignment</button>
          <button type="button" disabled={workflowBusy} onClick={() => runWorkflow(api.imuAlignmentReset, "Measurements reset")} className="w-full rounded-md border px-3 py-2 text-sm disabled:opacity-50">Reset measurements</button>
          <button type="button" disabled={workflowBusy} onClick={refreshWorkflow} className="w-full rounded-md border px-3 py-2 text-sm disabled:opacity-50">Refresh workflow status</button>
          {workflowMessage && <p role="status" aria-live="polite" className={`break-words text-xs ${workflowError ? "text-destructive" : "text-green-600"}`}>{workflowMessage}</p>}
        </div>

        {!propagationReady && (
          <div className="rounded-md border border-amber-500/40 bg-amber-500/5 p-2">
            <div className="text-xs font-semibold text-amber-500">
              PROVISIONAL - RELATIVE MOTION ONLY
            </div>
            <div className="mt-1 text-[11px] leading-snug text-muted-foreground">
              Celestial propagation requires both verified physical transforms.
              IMU yaw can drift because this sensor has no magnetometer.
            </div>
          </div>
        )}

        {imu ? (
          <>
            <div className="space-y-2">
              <div className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                Relative movement
              </div>
              <Reading label="Altitude" value={degrees(imu.altitude_relative_deg)} />
              <Reading label="Azimuth" value={degrees(imu.azimuth_relative_deg)} />
            </div>
            <div className="border-t pt-2 space-y-2">
              <div className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                Sensor attitude
              </div>
              <Reading label="Roll" value={degrees(imu.roll_deg)} />
              <Reading label="Pitch" value={degrees(imu.pitch_deg)} />
              <Reading label="Relative yaw" value={degrees(imu.yaw_relative_deg)} />
            </div>
            <div className="border-t pt-2 space-y-2">
              <div className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                Raw sensor readings (X / Y / Z)
              </div>
              <Reading label="Acceleration" value={axes(imu.raw_accel_g, "g")} />
              <Reading label="Gyroscope" value={axes(imu.raw_gyro_dps, "deg/s")} />
            </div>
            <div className="border-t pt-2 space-y-2">
              <Reading
                label="Reference"
                value={imu.reference_initialized ? "Initialized" : "Waiting"}
              />
              <Reading
                label="Sample age"
                value={Number.isFinite(imu.sample_age_s) ? `${imu.sample_age_s.toFixed(3)} s` : "?"}
              />
              <Reading
                label="Accel correction"
                value={imu.accel_correction_used ? "Active" : "Inactive"}
              />
              <Reading
                label="Yaw status"
                value={imu.yaw_drift_warning ? "Drift possible" : "Relative"}
              />
            </div>
          </>
        ) : (
          <div className="text-sm text-muted-foreground">
            Waiting for IMU data. Check that the sensor is connected and the
            serial reader is running.
          </div>
        )}
      </CardContent>
    </Card>
  );
}
