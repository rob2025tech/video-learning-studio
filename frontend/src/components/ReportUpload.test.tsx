import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ApiError, uploadReport } from "../api/client";
import type { UploadResponse } from "../api/types";
import ReportUpload from "./ReportUpload";

vi.mock("../api/client", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../api/client")>();
  return { ...actual, uploadReport: vi.fn() };
});

const uploadMock = vi.mocked(uploadReport);

const SESSION: UploadResponse = {
  report_id: "r-abc123",
  report_schema: "video-learning.report/v1",
  filename: "report.json",
  bytes_received: 3266,
  expires_in_seconds: 1800,
  video_duration_seconds: 24,
  transcription_status: "ok",
  transcription_available: true,
  corpus: {
    record_count: 7,
    searchable_record_count: 7,
    speech_record_count: 6,
    ocr_record_count: 1,
    segment_count: 1,
  },
};

function makeFile(name = "report.json"): File {
  return new File(["{}"], name, { type: "application/json" });
}

function selectFile(file: File): void {
  fireEvent.change(screen.getByLabelText("Report artifact (JSON)"), {
    target: { files: [file] },
  });
}

function factValue(label: string): string {
  return screen.getByText(label).nextElementSibling?.textContent ?? "";
}

beforeEach(() => {
  uploadMock.mockReset();
});

afterEach(cleanup);

describe("ReportUpload", () => {
  it("explains the input is a report artifact, not a video", () => {
    render(<ReportUpload />);
    expect(screen.getByText(/video-learning report/)).toBeTruthy();
    expect(screen.getByText(/never uploaded/)).toBeTruthy();
    expect(screen.getByRole("button", { name: "Upload report" })).toHaveProperty("disabled", true);
  });

  it("shows the selected filename and enables upload", () => {
    render(<ReportUpload />);
    selectFile(makeFile("my-report.json"));
    expect(screen.getByText("my-report.json")).toBeTruthy();
    expect(screen.getByRole("button", { name: "Upload report" })).toHaveProperty("disabled", false);
  });

  it("does not invent an extension restriction the backend lacks", () => {
    render(<ReportUpload />);
    // The backend validates content, not filename; a .txt name must still be uploadable.
    selectFile(makeFile("artifact.txt"));
    expect(screen.getByRole("button", { name: "Upload report" })).toHaveProperty("disabled", false);
  });

  it("uploads the selected file and renders the returned session", async () => {
    uploadMock.mockResolvedValue(SESSION);
    render(<ReportUpload />);
    const file = makeFile();
    selectFile(file);
    fireEvent.click(screen.getByRole("button", { name: "Upload report" }));

    await screen.findByText("Report loaded — session ready.");
    expect(uploadMock).toHaveBeenCalledWith(file);
    expect(screen.getByText("r-abc123")).toBeTruthy();
    expect(screen.getByText("report.json")).toBeTruthy();
    expect(screen.getByText("3.2 KB")).toBeTruthy();
    expect(screen.getByText("24 s")).toBeTruthy();
    expect(screen.getByText(/30 min/)).toBeTruthy();
    expect(factValue("Records")).toBe("7");
    expect(factValue("Searchable records")).toBe("7");
    expect(factValue("Speech records")).toBe("6");
    expect(factValue("OCR records")).toBe("1");
    expect(factValue("Segments")).toBe("1");
  });

  it("shows a loading state and prevents duplicate submission", async () => {
    let resolveUpload: (value: UploadResponse) => void = () => {};
    uploadMock.mockImplementation(
      () =>
        new Promise<UploadResponse>((resolve) => {
          resolveUpload = resolve;
        }),
    );
    render(<ReportUpload />);
    selectFile(makeFile());
    const button = screen.getByRole("button", { name: "Upload report" });
    fireEvent.click(button);

    await screen.findByText(/Uploading report.json/);
    expect(screen.getByRole("button", { name: "Upload report" })).toHaveProperty("disabled", true);
    expect(screen.getByLabelText("Report artifact (JSON)")).toHaveProperty("disabled", true);
    fireEvent.click(screen.getByRole("button", { name: "Upload report" }));
    expect(uploadMock).toHaveBeenCalledTimes(1);

    resolveUpload(SESSION);
    await screen.findByText("Report loaded — session ready.");
    expect(uploadMock).toHaveBeenCalledTimes(1);
  });

  it("renders an invalid-artifact error without echoing the server detail", async () => {
    uploadMock.mockRejectedValue(
      new ApiError(400, "invalid_report", "The report artifact is invalid."),
    );
    render(<ReportUpload />);
    selectFile(makeFile());
    fireEvent.click(screen.getByRole("button", { name: "Upload report" }));

    const alert = await screen.findByRole("alert");
    expect(alert.textContent).toContain("not a valid LearnLens report artifact");
    expect(alert.textContent).not.toContain("The report artifact is invalid.");
    expect(screen.getByRole("button", { name: "Try again" })).toHaveProperty("disabled", false);
  });

  it("retries the same file after an error", async () => {
    uploadMock.mockRejectedValueOnce(new ApiError(400, "invalid_report", "bad"));
    uploadMock.mockResolvedValueOnce(SESSION);
    render(<ReportUpload />);
    const file = makeFile();
    selectFile(file);
    fireEvent.click(screen.getByRole("button", { name: "Upload report" }));
    await screen.findByRole("alert");
    fireEvent.click(screen.getByRole("button", { name: "Try again" }));
    await screen.findByText("Report loaded — session ready.");
    expect(uploadMock).toHaveBeenCalledTimes(2);
    expect(uploadMock).toHaveBeenLastCalledWith(file);
  });

  it("renders a network failure with a start-the-API hint", async () => {
    uploadMock.mockRejectedValue(new ApiError(0, "network_error", "Could not reach"));
    render(<ReportUpload />);
    selectFile(makeFile());
    fireEvent.click(screen.getByRole("button", { name: "Upload report" }));
    const alert = await screen.findByRole("alert");
    expect(alert.textContent).toContain("uv run video-learning-web");
  });

  it("renders an unexpected server response without the raw body", async () => {
    uploadMock.mockRejectedValue(new ApiError(500, "http_500", "traceback secret"));
    render(<ReportUpload />);
    selectFile(makeFile());
    fireEvent.click(screen.getByRole("button", { name: "Upload report" }));
    const alert = await screen.findByRole("alert");
    expect(alert.textContent).toContain("unexpected response");
    expect(alert.textContent).not.toContain("traceback secret");
  });

  it("renders an oversized-report error", async () => {
    uploadMock.mockRejectedValue(new ApiError(413, "request_too_large", "too big"));
    render(<ReportUpload />);
    selectFile(makeFile());
    fireEvent.click(screen.getByRole("button", { name: "Upload report" }));
    const alert = await screen.findByRole("alert");
    expect(alert.textContent).toContain("upload limit");
  });

  it("lets the user choose another report after success", async () => {
    uploadMock.mockResolvedValue(SESSION);
    render(<ReportUpload />);
    selectFile(makeFile());
    fireEvent.click(screen.getByRole("button", { name: "Upload report" }));
    await screen.findByText("Report loaded — session ready.");

    fireEvent.click(screen.getByRole("button", { name: "Choose another report" }));
    expect(screen.getByLabelText("Report artifact (JSON)")).toBeTruthy();
    expect(screen.getByRole("button", { name: "Upload report" })).toHaveProperty("disabled", true);

    selectFile(makeFile("second.json"));
    fireEvent.click(screen.getByRole("button", { name: "Upload report" }));
    await screen.findByText("Report loaded — session ready.");
    expect(uploadMock).toHaveBeenCalledTimes(2);
  });

  it("accepts a drag-and-drop file", () => {
    render(<ReportUpload />);
    const dropzone = screen.getByText("Drag and drop, or choose a file.").closest(".dropzone");
    expect(dropzone).not.toBeNull();
    fireEvent.drop(dropzone as Element, { dataTransfer: { files: [makeFile("dropped.json")] } });
    expect(screen.getByText("dropped.json")).toBeTruthy();
    expect(screen.getByRole("button", { name: "Upload report" })).toHaveProperty("disabled", false);
  });

  it("shows the transcription-unavailable hint from the returned flag", async () => {
    uploadMock.mockResolvedValue({
      ...SESSION,
      transcription_status: "unavailable",
      transcription_available: false,
    });
    render(<ReportUpload />);
    selectFile(makeFile());
    fireEvent.click(screen.getByRole("button", { name: "Upload report" }));
    await screen.findByText("Report loaded — session ready.");
    expect(screen.getByText(/speech evidence unavailable/)).toBeTruthy();
  });

  it("handles an unknown duration without fabricating one", async () => {
    uploadMock.mockResolvedValue({ ...SESSION, video_duration_seconds: null });
    render(<ReportUpload />);
    selectFile(makeFile());
    fireEvent.click(screen.getByRole("button", { name: "Upload report" }));
    await screen.findByText("Report loaded — session ready.");
    expect(screen.getByText("Unknown")).toBeTruthy();
  });
});

describe("ReportUpload stale-upload guards and focus", () => {
  it("moves focus to the session summary heading on upload success", async () => {
    uploadMock.mockResolvedValue(SESSION);
    render(<ReportUpload />);
    selectFile(makeFile());
    fireEvent.click(screen.getByRole("button", { name: "Upload report" }));
    await screen.findByText("Report loaded — session ready.");
    const heading = screen.getByRole("heading", { name: "Report session" });
    expect(document.activeElement).toBe(heading);
    expect(heading).toHaveProperty("tabIndex", -1);
  });

  it("ignores a stale upload success on a still-mounted panel", async () => {
    const uploadSeq = { current: 0 };
    let resolveA: (value: UploadResponse) => void = () => {};
    uploadMock.mockImplementationOnce(
      () => new Promise<UploadResponse>((resolve) => { resolveA = resolve; }),
    );
    const onUploaded = vi.fn();
    render(<ReportUpload onUploaded={onUploaded} uploadSeq={uploadSeq} />);
    selectFile(makeFile("a.json"));
    fireEvent.click(screen.getByRole("button", { name: "Upload report" }));
    await screen.findByText(/Uploading a.json/);

    // A newer upload (possibly from a remounted panel) claims the sequence.
    uploadSeq.current += 1;
    resolveA(SESSION);
    await Promise.resolve();
    await Promise.resolve();
    await Promise.resolve();

    expect(screen.queryByText("Report loaded — session ready.")).toBeNull();
    expect(screen.getByText(/Uploading a.json/)).toBeTruthy();
    expect(onUploaded).not.toHaveBeenCalled();
  });

  it("ignores a stale upload failure instead of showing an error", async () => {
    const uploadSeq = { current: 0 };
    let rejectA: (reason: unknown) => void = () => {};
    uploadMock.mockImplementationOnce(
      () => new Promise<UploadResponse>((_, reject) => { rejectA = reject; }),
    );
    render(<ReportUpload uploadSeq={uploadSeq} />);
    selectFile(makeFile("a.json"));
    fireEvent.click(screen.getByRole("button", { name: "Upload report" }));
    await screen.findByText(/Uploading a.json/);

    uploadSeq.current += 1;
    rejectA(new ApiError(400, "invalid_report", "bad"));
    await Promise.resolve();
    await Promise.resolve();
    await Promise.resolve();

    expect(screen.queryByRole("alert")).toBeNull();
    expect(screen.getByText(/Uploading a.json/)).toBeTruthy();
  });
});