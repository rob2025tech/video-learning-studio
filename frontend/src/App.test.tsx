import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ApiError, searchReports, uploadReport } from "./api/client";
import type { SearchResponse, UploadResponse } from "./api/types";
import App from "./App";

vi.mock("./api/client", async (importOriginal) => {
  const actual = await importOriginal<typeof import("./api/client")>();
  return { ...actual, uploadReport: vi.fn(), searchReports: vi.fn() };
});

const uploadMock = vi.mocked(uploadReport);
const searchMock = vi.mocked(searchReports);

const createObjectURL = vi.fn();
const revokeObjectURL = vi.fn();
let urlCounter = 0;

// jsdom ships no matchMedia; seeking queries prefers-reduced-motion through it.
function mockMatchMedia(prefersReducedMotion: boolean): void {
  Object.defineProperty(window, "matchMedia", {
    writable: true,
    configurable: true,
    value: vi.fn().mockImplementation((query: string) => ({
      matches: prefersReducedMotion,
      media: query,
      onchange: null,
      addListener: vi.fn(),
      removeListener: vi.fn(),
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
      dispatchEvent: vi.fn(),
    })),
  });
}

const SESSION: UploadResponse = {
  report_id: "r-abc123",
  report_schema: "video-learning.report/v1",
  filename: "clip-report.json",
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

const RESPONSE: SearchResponse = {
  schema: "video-learning.search/v1",
  query: "index",
  status: "ok",
  source: "clip.mov",
  source_is_redacted: true,
  study_evidence_schema: "video-learning.study-evidence/v2",
  transcription_status: "ok",
  transcription_available: true,
  video_duration_seconds: 24,
  limit: 5,
  results: [
    {
      evidence_id: "speech:0",
      source_type: "speech",
      text: "the index ranks record 0",
      start_seconds: 0,
      end_seconds: 3,
      segment_id: 0,
      score: 0.0645,
      matched_terms: ["index"],
      matched_spans: [{ term: "index", surface: "index", start: 4, end: 9 }],
    },
  ],
  result_count: 1,
  coverage: {
    record_count: 7,
    searchable_record_count: 7,
    speech_record_count: 6,
    ocr_record_count: 1,
    attributed_record_count: 7,
    unattributed_record_count: 0,
    query_token_count: 1,
    meaningful_query_term_count: 1,
    matched_record_count: 1,
    returned_record_count: 1,
  },
};

beforeEach(() => {
  uploadMock.mockReset();
  searchMock.mockReset();
  urlCounter = 0;
  createObjectURL.mockReset();
  revokeObjectURL.mockReset();
  createObjectURL.mockImplementation((file: File) => `blob:mock-${file.name}-${++urlCounter}`);
  Object.defineProperty(URL, "createObjectURL", {
    value: createObjectURL,
    configurable: true,
    writable: true,
  });
  Object.defineProperty(URL, "revokeObjectURL", {
    value: revokeObjectURL,
    configurable: true,
    writable: true,
  });
  vi.spyOn(HTMLMediaElement.prototype, "play").mockImplementation(() => Promise.resolve());
  vi.spyOn(HTMLMediaElement.prototype, "pause").mockImplementation(() => {});
  vi.spyOn(HTMLMediaElement.prototype, "canPlayType").mockReturnValue("probably");
  Element.prototype.scrollIntoView = vi.fn();
  mockMatchMedia(false);
});

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

describe("App shell", () => {
  it("renders the application title", () => {
    render(<App />);
    expect(screen.getByRole("heading", { level: 1 }).textContent).toBe("LearnLens");
  });

  it("renders the tagline", () => {
    render(<App />);
    expect(screen.getByText("Local-first video study evidence")).toBeTruthy();
  });

  it("marks Workspace as the current page in the primary navigation", () => {
    render(<App />);
    const nav = screen.getByRole("navigation", { name: "Primary" });
    const current = screen.getByRole("button", { name: "Workspace" });
    expect(nav.contains(current)).toBe(true);
    expect(current.getAttribute("aria-current")).toBe("page");
  });

  it("keeps Evidence search gated until a report is uploaded", () => {
    render(<App />);
    const item = screen.getByText("Evidence search");
    expect(item.getAttribute("aria-disabled")).toBe("true");
    expect(item.closest("button")).toBeNull();
    expect(item.closest("a")).toBeNull();
  });

  it("introduces the workspace inside a main landmark", () => {
    render(<App />);
    const main = screen.getByRole("main");
    const heading = screen.getByRole("heading", {
      level: 2,
      name: "Video learning, grounded in evidence",
    });
    expect(main.contains(heading)).toBe(true);
  });

  it("switches to the upload view and back via the navigation", () => {
    render(<App />);
    fireEvent.click(screen.getByRole("button", { name: "Report upload" }));
    expect(
      screen.getByRole("heading", { level: 2, name: "Upload a report artifact" }),
    ).toBeTruthy();
    expect(screen.getByRole("button", { name: "Report upload" }).getAttribute("aria-current")).toBe(
      "page",
    );
    expect(screen.getByRole("button", { name: "Workspace" }).getAttribute("aria-current")).toBeNull();

    fireEvent.click(screen.getByRole("button", { name: "Workspace" }));
    expect(
      screen.getByRole("heading", { level: 2, name: "Video learning, grounded in evidence" }),
    ).toBeTruthy();
  });
});

describe("App session flow", () => {
  it("enables Evidence search after a successful upload and opens it", async () => {
    uploadMock.mockResolvedValue(SESSION);
    render(<App />);

    fireEvent.click(screen.getByRole("button", { name: "Report upload" }));
    fireEvent.change(screen.getByLabelText("Report artifact (JSON)"), {
      target: { files: [new File(["{}"], "clip-report.json", { type: "application/json" })] },
    });
    fireEvent.click(screen.getByRole("button", { name: "Upload report" }));
    await screen.findByText("Report loaded — session ready.");

    const searchNav = screen.getByRole("button", { name: "Evidence search" });
    expect(searchNav.getAttribute("aria-disabled")).toBeNull();
    fireEvent.click(searchNav);
    expect(searchNav.getAttribute("aria-current")).toBe("page");
    expect(screen.getByRole("heading", { level: 2, name: "Evidence search" })).toBeTruthy();
    expect(screen.getByText("clip-report.json")).toBeTruthy();
  });

  it("runs a query against the uploaded session and shows evidence", async () => {
    uploadMock.mockResolvedValue(SESSION);
    searchMock.mockResolvedValue(RESPONSE);
    render(<App />);

    fireEvent.click(screen.getByRole("button", { name: "Report upload" }));
    fireEvent.change(screen.getByLabelText("Report artifact (JSON)"), {
      target: { files: [new File(["{}"], "clip-report.json", { type: "application/json" })] },
    });
    fireEvent.click(screen.getByRole("button", { name: "Upload report" }));
    await screen.findByText("Report loaded — session ready.");

    fireEvent.click(screen.getByRole("button", { name: "Evidence search" }));
    fireEvent.change(screen.getByLabelText("Search query"), { target: { value: "index" } });
    fireEvent.click(screen.getByRole("button", { name: "Search" }));
    await screen.findByText("SPEECH");
    expect(searchMock).toHaveBeenCalledWith({ report_id: "r-abc123", query: "index", limit: 5 });
    expect(screen.getByText("00:00:00.0 → 00:00:03.0")).toBeTruthy();
  });

  it("references the loaded report in the workspace intro", async () => {
    uploadMock.mockResolvedValue(SESSION);
    render(<App />);

    fireEvent.click(screen.getByRole("button", { name: "Report upload" }));
    fireEvent.change(screen.getByLabelText("Report artifact (JSON)"), {
      target: { files: [new File(["{}"], "clip-report.json", { type: "application/json" })] },
    });
    fireEvent.click(screen.getByRole("button", { name: "Upload report" }));
    await screen.findByText("Report loaded — session ready.");

    fireEvent.click(screen.getByRole("button", { name: "Workspace" }));
    expect(screen.getByText(/is loaded\. Open Evidence search/)).toBeTruthy();
    expect(screen.getByText(/its 7 records/)).toBeTruthy();
  });
});

describe("App video session lifecycle", () => {
  function uploadReportFile(name = "clip-report.json"): void {
    fireEvent.click(screen.getByRole("button", { name: "Report upload" }));
    fireEvent.change(screen.getByLabelText("Report artifact (JSON)"), {
      target: { files: [new File(["{}"], name, { type: "application/json" })] },
    });
    fireEvent.click(screen.getByRole("button", { name: "Upload report" }));
  }

  function selectLocalVideo(name = "clip.mp4"): void {
    fireEvent.change(screen.getByLabelText("Original video (local file)"), {
      target: { files: [new File(["video-bytes"], name, { type: "video/mp4" })] },
    });
  }

  it("preserves the selected video across view switches", async () => {
    uploadMock.mockResolvedValue(SESSION);
    render(<App />);
    uploadReportFile();
    await screen.findByText("Report loaded — session ready.");
    fireEvent.click(screen.getByRole("button", { name: "Evidence search" }));
    selectLocalVideo("clip.mp4");
    expect(screen.getByText("clip.mp4")).toBeTruthy();

    fireEvent.click(screen.getByRole("button", { name: "Workspace" }));
    fireEvent.click(screen.getByRole("button", { name: "Evidence search" }));
    expect(screen.getByText("clip.mp4")).toBeTruthy();
  });

  it("clears the selected video when a new report is uploaded", async () => {
    uploadMock
      .mockResolvedValueOnce(SESSION)
      .mockResolvedValueOnce({ ...SESSION, report_id: "r-second", filename: "second.json" });
    render(<App />);
    uploadReportFile();
    await screen.findByText("Report loaded — session ready.");
    fireEvent.click(screen.getByRole("button", { name: "Evidence search" }));
    selectLocalVideo("clip.mp4");
    expect(screen.getByText("clip.mp4")).toBeTruthy();

    uploadReportFile("second.json");
    await screen.findByText("Report loaded — session ready.");
    fireEvent.click(screen.getByRole("button", { name: "Evidence search" }));
    expect(screen.getByText("No video selected.")).toBeTruthy();
    expect(screen.getByText("second.json")).toBeTruthy();
  });

  it("keeps the active session and video when an upload fails", async () => {
    uploadMock.mockResolvedValueOnce(SESSION);
    render(<App />);
    uploadReportFile();
    await screen.findByText("Report loaded — session ready.");
    fireEvent.click(screen.getByRole("button", { name: "Evidence search" }));
    selectLocalVideo("clip.mp4");
    expect(screen.getByText("clip.mp4")).toBeTruthy();

    uploadMock.mockRejectedValueOnce(new ApiError(400, "invalid_report", "bad"));
    uploadReportFile("bad.json");
    await screen.findByRole("alert");

    fireEvent.click(screen.getByRole("button", { name: "Evidence search" }));
    expect(screen.getByText("clip.mp4")).toBeTruthy();
    expect(screen.getByText("clip-report.json")).toBeTruthy();
  });

  it("keeps search functional with no video selected", async () => {
    uploadMock.mockResolvedValue(SESSION);
    searchMock.mockResolvedValue(RESPONSE);
    render(<App />);
    uploadReportFile();
    await screen.findByText("Report loaded — session ready.");
    fireEvent.click(screen.getByRole("button", { name: "Evidence search" }));
    expect(screen.getByText("No video selected.")).toBeTruthy();

    fireEvent.change(screen.getByLabelText("Search query"), { target: { value: "index" } });
    fireEvent.click(screen.getByRole("button", { name: "Search" }));
    await screen.findByText("SPEECH");
    expect(searchMock).toHaveBeenCalledWith({ report_id: "r-abc123", query: "index", limit: 5 });
  });
});

describe("App view coherence and async safety", () => {
  function selectReport(name: string): void {
    fireEvent.change(screen.getByLabelText("Report artifact (JSON)"), {
      target: { files: [new File(["{}"], name, { type: "application/json" })] },
    });
  }

  function selectVideo(name = "clip.mp4"): void {
    fireEvent.change(screen.getByLabelText("Original video (local file)"), {
      target: { files: [new File(["video-bytes"], name, { type: "video/mp4" })] },
    });
  }

  async function uploadAndOpenSearch(): Promise<void> {
    fireEvent.click(screen.getByRole("button", { name: "Report upload" }));
    selectReport("clip-report.json");
    fireEvent.click(screen.getByRole("button", { name: "Upload report" }));
    await screen.findByText("Report loaded — session ready.");
    fireEvent.click(screen.getByRole("button", { name: "Evidence search" }));
  }

  it("keeps the newer session when an older upload completes after a remount", async () => {
    let resolveA: (value: UploadResponse) => void = () => {};
    let resolveB: (value: UploadResponse) => void = () => {};
    uploadMock
      .mockImplementationOnce(
        () => new Promise<UploadResponse>((resolve) => { resolveA = resolve; }),
      )
      .mockImplementationOnce(
        () => new Promise<UploadResponse>((resolve) => { resolveB = resolve; }),
      );
    render(<App />);

    fireEvent.click(screen.getByRole("button", { name: "Report upload" }));
    selectReport("a.json");
    fireEvent.click(screen.getByRole("button", { name: "Upload report" }));
    await screen.findByText(/Uploading a.json/);

    // Leave and return: the upload panel remounts while A is still in flight.
    fireEvent.click(screen.getByRole("button", { name: "Workspace" }));
    fireEvent.click(screen.getByRole("button", { name: "Report upload" }));
    selectReport("b.json");
    fireEvent.click(screen.getByRole("button", { name: "Upload report" }));
    await screen.findByText(/Uploading b.json/);

    resolveB({ ...SESSION, report_id: "r-b", filename: "b.json" });
    await screen.findByText("Report loaded — session ready.");
    expect(screen.getByText("r-b")).toBeTruthy();

    resolveA({ ...SESSION, report_id: "r-a", filename: "a.json" });
    await Promise.resolve();
    await Promise.resolve();
    await Promise.resolve();

    expect(screen.getByText("r-b")).toBeTruthy();
    expect(screen.queryByText("r-a")).toBeNull();
  });

  it("preserves the query and results across view switches", async () => {
    uploadMock.mockResolvedValue(SESSION);
    searchMock.mockResolvedValue(RESPONSE);
    render(<App />);
    await uploadAndOpenSearch();

    fireEvent.change(screen.getByLabelText("Search query"), { target: { value: "index" } });
    fireEvent.click(screen.getByRole("button", { name: "Search" }));
    await screen.findByText("SPEECH");
    expect(screen.getByLabelText("Search query")).toHaveProperty("value", "index");

    fireEvent.click(screen.getByRole("button", { name: "Workspace" }));
    fireEvent.click(screen.getByRole("button", { name: "Evidence search" }));
    expect(screen.getByLabelText("Search query")).toHaveProperty("value", "index");
    expect(screen.getByText("SPEECH")).toBeTruthy();
    expect(screen.getByText("00:00:00.0 → 00:00:03.0")).toBeTruthy();
  });

  it("preserves playback position and selected evidence across view switches", async () => {
    uploadMock.mockResolvedValue(SESSION);
    searchMock.mockResolvedValue(RESPONSE);
    const { container } = render(<App />);
    await uploadAndOpenSearch();

    selectVideo();
    const video = container.querySelector("video");
    expect(video).not.toBeNull();
    Object.defineProperty(video, "duration", { value: 24, configurable: true });
    fireEvent(video as HTMLVideoElement, new Event("loadedmetadata"));
    fireEvent(video as HTMLVideoElement, new Event("canplay"));

    fireEvent.change(screen.getByLabelText("Search query"), { target: { value: "index" } });
    fireEvent.click(screen.getByRole("button", { name: "Search" }));
    await screen.findByText("SPEECH");
    fireEvent.click(screen.getByRole("button", { name: /speech evidence/ }));
    expect(screen.getByText("Selected")).toBeTruthy();
    expect(screen.getByText("Jumped to 00:00:00.0")).toBeTruthy();
    (video as HTMLVideoElement).currentTime = 7.5;

    fireEvent.click(screen.getByRole("button", { name: "Workspace" }));
    fireEvent.click(screen.getByRole("button", { name: "Evidence search" }));
    expect((video as HTMLVideoElement).currentTime).toBe(7.5);
    expect(screen.getByText("Selected")).toBeTruthy();
    expect(screen.getByText("Jumped to 00:00:00.0")).toBeTruthy();
    expect(
      screen.getByRole("button", { name: /speech evidence/ }).getAttribute("aria-current"),
    ).toBe("true");
  });

  it("does not mark a result selected when the player cannot seek (error phase)", async () => {
    uploadMock.mockResolvedValue(SESSION);
    searchMock.mockResolvedValue(RESPONSE);
    const { container } = render(<App />);
    await uploadAndOpenSearch();

    selectVideo();
    fireEvent(container.querySelector("video") as HTMLVideoElement, new Event("error"));

    fireEvent.change(screen.getByLabelText("Search query"), { target: { value: "index" } });
    fireEvent.click(screen.getByRole("button", { name: "Search" }));
    await screen.findByText("SPEECH");
    fireEvent.click(screen.getByRole("button", { name: /speech evidence/ }));

    expect(screen.queryByText("Selected")).toBeNull();
    expect(screen.queryByText(/Jumped to/)).toBeNull();
    expect(
      screen.getByRole("button", { name: /speech evidence/ }).getAttribute("aria-current"),
    ).toBeNull();
  });
});