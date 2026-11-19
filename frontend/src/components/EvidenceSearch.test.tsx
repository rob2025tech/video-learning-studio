import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ApiError, searchReports } from "../api/client";
import type { SearchResponse, UploadResponse } from "../api/types";
import EvidenceSearch from "./EvidenceSearch";

vi.mock("../api/client", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../api/client")>();
  return { ...actual, searchReports: vi.fn() };
});

const searchMock = vi.mocked(searchReports);

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
    {
      evidence_id: "ocr:0",
      source_type: "ocr",
      text: "build a search index locally",
      start_seconds: 0.5,
      end_seconds: null,
      segment_id: 0,
      score: 0.0645,
      matched_terms: ["index"],
      matched_spans: [{ term: "index", surface: "index", start: 15, end: 20 }],
    },
  ],
  result_count: 2,
  coverage: {
    record_count: 7,
    searchable_record_count: 7,
    speech_record_count: 6,
    ocr_record_count: 1,
    attributed_record_count: 7,
    unattributed_record_count: 0,
    query_token_count: 1,
    meaningful_query_term_count: 1,
    matched_record_count: 7,
    returned_record_count: 2,
  },
};

function emptyResponse(overrides: Partial<SearchResponse>): SearchResponse {
  return { ...RESPONSE, results: [], result_count: 0, ...overrides };
}

const onNavigateUpload = vi.fn();

function renderSearch(session: UploadResponse = SESSION) {
  return render(<EvidenceSearch session={session} onNavigateUpload={onNavigateUpload} />);
}

function typeQuery(value: string): void {
  fireEvent.change(screen.getByLabelText("Search query"), { target: { value } });
}

beforeEach(() => {
  searchMock.mockReset();
  onNavigateUpload.mockReset();
});

afterEach(cleanup);

describe("EvidenceSearch", () => {
  it("shows the active report filename and disables search for an empty query", () => {
    renderSearch();
    expect(screen.getByText("clip-report.json")).toBeTruthy();
    expect(screen.getByRole("button", { name: "Search" })).toHaveProperty("disabled", true);
  });

  it("blocks a whitespace-only query with a hint", () => {
    renderSearch();
    typeQuery("   ");
    expect(screen.getByRole("button", { name: "Search" })).toHaveProperty("disabled", true);
    expect(screen.getByText(/at least one non-space character/)).toBeTruthy();
  });

  it("sends the trimmed query with the session id and default limit", async () => {
    searchMock.mockResolvedValue(RESPONSE);
    renderSearch();
    typeQuery("  index  ");
    fireEvent.click(screen.getByRole("button", { name: "Search" }));
    await screen.findByText("SPEECH");
    expect(searchMock).toHaveBeenCalledWith({ report_id: "r-abc123", query: "index", limit: 5 });
  });

  it("renders evidence with timestamps, highlights, and matched terms", async () => {
    searchMock.mockResolvedValue(RESPONSE);
    const { container } = renderSearch();
    typeQuery("index");
    fireEvent.click(screen.getByRole("button", { name: "Search" }));
    await screen.findByText("SPEECH");
    expect(screen.getByText("OCR")).toBeTruthy();
    expect(screen.getByText("00:00:00.0 → 00:00:03.0")).toBeTruthy();
    expect(screen.getByText("00:00:00.5")).toBeTruthy();
    expect(container.querySelectorAll("mark")).toHaveLength(2);
    expect(screen.getAllByText("matched index")).toHaveLength(2);
    expect(container.textContent).toContain("7 matching records of 7 searchable");
  });

  it("honors the selected results limit", async () => {
    searchMock.mockResolvedValue(RESPONSE);
    renderSearch();
    fireEvent.change(screen.getByLabelText("Results limit"), { target: { value: "10" } });
    typeQuery("index");
    fireEvent.click(screen.getByRole("button", { name: "Search" }));
    await screen.findByText("SPEECH");
    expect(searchMock).toHaveBeenCalledWith({ report_id: "r-abc123", query: "index", limit: 10 });
  });

  it("shows a loading state and prevents duplicate submission", async () => {
    let resolveSearch: (value: SearchResponse) => void = () => {};
    searchMock.mockImplementation(
      () =>
        new Promise<SearchResponse>((resolve) => {
          resolveSearch = resolve;
        }),
    );
    renderSearch();
    typeQuery("index");
    fireEvent.click(screen.getByRole("button", { name: "Search" }));
    await screen.findByText(/Searching clip-report.json/);
    expect(screen.getByRole("button", { name: "Searching…" })).toHaveProperty("disabled", true);
    fireEvent.click(screen.getByRole("button", { name: "Searching…" }));
    expect(searchMock).toHaveBeenCalledTimes(1);
    resolveSearch(RESPONSE);
    await screen.findByText("SPEECH");
    expect(searchMock).toHaveBeenCalledTimes(1);
  });

  it("explains a no-confident-match empty answer from coverage", async () => {
    searchMock.mockResolvedValue(
      emptyResponse({
        status: "no_confident_match",
        coverage: { ...RESPONSE.coverage, matched_record_count: 0, returned_record_count: 0 },
      }),
    );
    renderSearch();
    typeQuery("zzz");
    fireEvent.click(screen.getByRole("button", { name: "Search" }));
    expect(await screen.findByText(/None of the 1 meaningful query terms/)).toBeTruthy();
  });

  it("explains a no-evidence artifact", async () => {
    searchMock.mockResolvedValue(
      emptyResponse({
        status: "no_evidence",
        coverage: {
          ...RESPONSE.coverage,
          record_count: 3,
          searchable_record_count: 0,
          matched_record_count: 0,
          returned_record_count: 0,
        },
      }),
    );
    renderSearch();
    typeQuery("index");
    fireEvent.click(screen.getByRole("button", { name: "Search" }));
    expect(await screen.findByText(/Nothing in this artifact is searchable/)).toBeTruthy();
  });

  it("explains an all-stopword query", async () => {
    searchMock.mockResolvedValue(
      emptyResponse({
        status: "no_confident_match",
        coverage: {
          ...RESPONSE.coverage,
          query_token_count: 3,
          meaningful_query_term_count: 0,
          matched_record_count: 0,
          returned_record_count: 0,
        },
      }),
    );
    renderSearch();
    typeQuery("the and of");
    fireEvent.click(screen.getByRole("button", { name: "Search" }));
    expect(await screen.findByText(/none of them a meaningful term/)).toBeTruthy();
  });

  it("explains a non-positive limit that withheld matches", async () => {
    searchMock.mockResolvedValue(
      emptyResponse({
        limit: 0,
        coverage: { ...RESPONSE.coverage, matched_record_count: 7, returned_record_count: 0 },
      }),
    );
    renderSearch();
    typeQuery("index");
    fireEvent.click(screen.getByRole("button", { name: "Search" }));
    expect(await screen.findByText(/the limit \(0\) asked for none/)).toBeTruthy();
  });

  it("notes when the limit capped the answer", async () => {
    searchMock.mockResolvedValue(RESPONSE);
    renderSearch();
    typeQuery("index");
    fireEvent.click(screen.getByRole("button", { name: "Search" }));
    expect(await screen.findByText(/Showing 2 of 7 matching records/)).toBeTruthy();
  });

  it("offers re-upload on an unknown report and navigates", async () => {
    searchMock.mockRejectedValue(new ApiError(404, "unknown_report", "gone"));
    renderSearch();
    typeQuery("index");
    fireEvent.click(screen.getByRole("button", { name: "Search" }));
    const alert = await screen.findByRole("alert");
    expect(alert.textContent).toContain("no longer exists");
    fireEvent.click(screen.getByRole("button", { name: "Go to report upload" }));
    expect(onNavigateUpload).toHaveBeenCalledTimes(1);
  });

  it("offers re-upload on an expired report", async () => {
    searchMock.mockRejectedValue(new ApiError(404, "report_expired", "expired"));
    renderSearch();
    typeQuery("index");
    fireEvent.click(screen.getByRole("button", { name: "Search" }));
    const alert = await screen.findByRole("alert");
    expect(alert.textContent).toContain("expired");
    expect(screen.getByRole("button", { name: "Go to report upload" })).toBeTruthy();
  });

  it("renders a network failure without a re-upload button", async () => {
    searchMock.mockRejectedValue(new ApiError(0, "network_error", "down"));
    renderSearch();
    typeQuery("index");
    fireEvent.click(screen.getByRole("button", { name: "Search" }));
    const alert = await screen.findByRole("alert");
    expect(alert.textContent).toContain("uv run video-learning-web");
    expect(screen.queryByRole("button", { name: "Go to report upload" })).toBeNull();
  });

  it("does not highlight when a span breaks the surface invariant", async () => {
    searchMock.mockResolvedValue({
      ...RESPONSE,
      results: [
        {
          ...RESPONSE.results[0],
          matched_spans: [{ term: "index", surface: "WRONG", start: 4, end: 9 }],
        },
      ],
      result_count: 1,
    });
    const { container } = renderSearch();
    typeQuery("index");
    fireEvent.click(screen.getByRole("button", { name: "Search" }));
    await screen.findByText("SPEECH");
    expect(container.querySelectorAll("mark")).toHaveLength(0);
    expect(container.textContent).toContain("the index ranks record 0");
  });

  it("resets query and results when the active report changes", async () => {
    searchMock.mockResolvedValue(RESPONSE);
    const { rerender } = renderSearch();
    typeQuery("index");
    fireEvent.click(screen.getByRole("button", { name: "Search" }));
    await screen.findByText("SPEECH");

    rerender(
      <EvidenceSearch
        session={{ ...SESSION, report_id: "r-new", filename: "other.json" }}
        onNavigateUpload={onNavigateUpload}
      />,
    );
    expect(screen.getByLabelText("Search query")).toHaveProperty("value", "");
    expect(screen.queryByText("SPEECH")).toBeNull();
    expect(screen.getByText("other.json")).toBeTruthy();
  });
});

describe("EvidenceSearch timestamp seeking", () => {
  async function searchWith({
    hasVideo = false,
    selectedEvidenceId = null,
  }: {
    hasVideo?: boolean;
    selectedEvidenceId?: string | null;
  }) {
    searchMock.mockResolvedValue(RESPONSE);
    const onSeek = vi.fn();
    render(
      <EvidenceSearch
        session={SESSION}
        onNavigateUpload={onNavigateUpload}
        onSeek={onSeek}
        hasVideo={hasVideo}
        selectedEvidenceId={selectedEvidenceId}
      />,
    );
    typeQuery("index");
    fireEvent.click(screen.getByRole("button", { name: "Search" }));
    await screen.findByText("SPEECH");
    return onSeek;
  }

  it("activates a speech result through a semantic button and seeks to its start", async () => {
    const onSeek = await searchWith({ hasVideo: true });
    const button = screen.getByRole("button", { name: /speech evidence/ });
    expect(button.tagName).toBe("BUTTON");
    fireEvent.click(button);
    expect(onSeek).toHaveBeenCalledWith(0, "speech:0");
  });

  it("seeks an OCR point result to its start_seconds", async () => {
    const onSeek = await searchWith({ hasVideo: true });
    fireEvent.click(screen.getByRole("button", { name: /OCR evidence/ }));
    expect(onSeek).toHaveBeenCalledWith(0.5, "ocr:0");
  });

  it("gives the selected evidence result a clear indication", async () => {
    await searchWith({ hasVideo: true, selectedEvidenceId: "speech:0" });
    expect(
      screen.getByRole("button", { name: /speech evidence/ }).getAttribute("aria-current"),
    ).toBe("true");
    expect(screen.getByText("Selected")).toBeTruthy();
  });

  it("explains a video is needed when none is selected, without throwing", async () => {
    const onSeek = await searchWith({ hasVideo: false });
    expect(screen.getByText(/Timestamp navigation needs a local video/)).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: /speech evidence/ }));
    expect(screen.getByText("Select a local video to jump to this timestamp.")).toBeTruthy();
    expect(onSeek).toHaveBeenCalledWith(0, "speech:0");
  });

  it("clears the stale select-a-video notice once a video is selected", async () => {
    searchMock.mockResolvedValue(RESPONSE);
    const { rerender } = render(
      <EvidenceSearch session={SESSION} onNavigateUpload={onNavigateUpload} hasVideo={false} />,
    );
    typeQuery("index");
    fireEvent.click(screen.getByRole("button", { name: "Search" }));
    await screen.findByText("SPEECH");
    fireEvent.click(screen.getByRole("button", { name: /speech evidence/ }));
    expect(screen.getByText("Select a local video to jump to this timestamp.")).toBeTruthy();

    rerender(
      <EvidenceSearch session={SESSION} onNavigateUpload={onNavigateUpload} hasVideo={true} />,
    );
    expect(screen.queryByText("Select a local video to jump to this timestamp.")).toBeNull();
    expect(screen.queryByText(/Timestamp navigation needs a local video/)).toBeNull();
  });
});

describe("EvidenceSearch async safety and focus", () => {
  it("does not let a stale search response overwrite newer results", async () => {
    let resolveOld: (value: SearchResponse) => void = () => {};
    let resolveNew: (value: SearchResponse) => void = () => {};
    searchMock
      .mockImplementationOnce(
        () => new Promise<SearchResponse>((resolve) => { resolveOld = resolve; }),
      )
      .mockImplementationOnce(
        () => new Promise<SearchResponse>((resolve) => { resolveNew = resolve; }),
      );
    const { rerender } = renderSearch();
    typeQuery("index");
    fireEvent.click(screen.getByRole("button", { name: "Search" }));
    await screen.findByText(/Searching clip-report.json/);

    // A report change invalidates the in-flight search and resets the panel; the
    // old response is now stale relative to the next search.
    rerender(
      <EvidenceSearch
        session={{ ...SESSION, report_id: "r-new", filename: "new.json" }}
        onNavigateUpload={onNavigateUpload}
      />,
    );
    typeQuery("index");
    fireEvent.click(screen.getByRole("button", { name: "Search" }));
    await screen.findByText(/Searching new.json/);

    const newer: SearchResponse = {
      ...RESPONSE,
      results: [
        {
          ...RESPONSE.results[0],
          evidence_id: "speech:new",
          text: "newer evidence result",
          matched_spans: [],
          matched_terms: [],
        },
      ],
      result_count: 1,
      coverage: { ...RESPONSE.coverage, matched_record_count: 1, returned_record_count: 1 },
    };
    resolveNew(newer);
    await screen.findByText("newer evidence result");

    resolveOld(RESPONSE);
    await Promise.resolve();
    await Promise.resolve();
    await Promise.resolve();

    expect(screen.getByText("newer evidence result")).toBeTruthy();
    expect(screen.queryByText("the index ranks record 0")).toBeNull();
  });

  it("moves focus to the results summary when a search completes", async () => {
    searchMock.mockResolvedValue(RESPONSE);
    const { container } = renderSearch();
    typeQuery("index");
    fireEvent.click(screen.getByRole("button", { name: "Search" }));
    await screen.findByText("SPEECH");
    const summary = container.querySelector(".results-summary");
    expect(summary).not.toBeNull();
    expect(summary?.getAttribute("role")).toBe("status");
    expect(document.activeElement).toBe(summary);
  });

  it("does not steal focus when results complete in a hidden view", async () => {
    searchMock.mockResolvedValue(RESPONSE);
    const { container } = render(
      <div hidden>
        <EvidenceSearch session={SESSION} onNavigateUpload={onNavigateUpload} />
      </div>,
    );
    typeQuery("index");
    fireEvent.click(screen.getByRole("button", { name: "Search", hidden: true }));
    await screen.findByText("SPEECH");
    const summary = container.querySelector(".results-summary");
    expect(summary).not.toBeNull();
    expect(document.activeElement).not.toBe(summary);
  });
});