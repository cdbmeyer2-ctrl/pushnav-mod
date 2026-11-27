import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { ImuStatus } from "@/components/ImuStatus";
import type { EnginePayload } from "@/lib/types";

const availableState = {
  imu: {
    available: true,
    altitude_relative_deg: 0,
    azimuth_relative_deg: 0,
    roll_deg: 0,
    pitch_deg: 0,
    yaw_relative_deg: 0,
    raw_accel_g: [0, 0, 1],
    raw_gyro_dps: [0, 0, 0],
    reference_initialized: true,
    sample_age_s: 0.01,
    accel_correction_used: true,
    yaw_drift_warning: false,
  },
} as unknown as EnginePayload;

const unavailableState = {
  imu: { available: false },
} as unknown as EnginePayload;

afterEach(() => {
  vi.restoreAllMocks();
});

describe("ImuStatus stationary calibration", () => {
  it("submits calibration and reports success", async () => {
    const fetchMock = vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(null, { status: 204 }),
    );

    render(<ImuStatus state={availableState} />);
    fireEvent.click(screen.getByRole("button", { name: "Calibrate gyroscope" }));

    await waitFor(() => {
      expect(fetchMock).toHaveBeenCalledWith(
        "/api/imu/calibrate-stationary",
        { method: "POST" },
      );
    });
    expect(await screen.findByText("Gyroscope calibration saved successfully.")).toHaveTextContent(
      "Gyroscope calibration saved successfully.",
    );
  });

  it("reports API errors without claiming calibration succeeded", async () => {
    vi.spyOn(globalThis, "fetch")
      .mockResolvedValueOnce(new Response("{}", { status: 200 }))
      .mockResolvedValueOnce(
        new Response("IMU samples are stale", { status: 500 }),
      );

    render(<ImuStatus state={availableState} />);
    fireEvent.click(screen.getByRole("button", { name: "Calibrate gyroscope" }));

    const errorMessage = await screen.findByText(
      /POST \/api\/imu\/calibrate-stationary/,
    );
    expect(errorMessage).toHaveClass("text-destructive");
    expect(
      screen.queryByText("Gyroscope calibration saved successfully."),
    ).not.toBeInTheDocument();
  });
  it("disables calibration when IMU data is unavailable", () => {
    render(<ImuStatus state={unavailableState} />);
    expect(screen.getByRole("button", { name: "Calibrate gyroscope" })).toBeDisabled();
  });
});



