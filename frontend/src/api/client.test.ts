import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ApiError, getHealth, searchReports, uploadReport } from "./client";
import type { HealthResponse, SearchResponse, UploadResponse } from "./types";

const fetchMock = vi.fn<typeof fetch>();

beforeEach(() => {
  vi.stubGlobal("fetch", fetchMock);
});

afterEach(() => {
  vi.unstubAllGlobals();
  fetchMock.mockReset();
});

function jsonResponse(status: number, body: unknown): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

const HEALTH: HealthResponse = {
  status: "ok",
  version: "0.1.0",
  report_schema: "video-learning.report/v1",
  search_schema: "video-learning.search/v1",
  active_report_sessions: 0,
  max_report_sessions: 8,
  max_report_bytes: 8388608,
};

const UPLOAD: UploadResponse = {
  report_id: "r-1",
  report_schema: "video-learning.report/v1",
  filename: "report.json",
  bytes_received: 1234,
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

const SEARCH: SearchResponse = {
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
      evidence_id: "speech-0",
      source_type: "speech",
      text: "the index ranks record 0",
      start_seconds: 0,
      end_seconds: 3,
      segment_id: 0,
      score: 1.25,
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
    matched_record_count: 6,
    returned_record_count: 1,
  },
};

describe("getHealth", () => {
  it("GETs the relative health path and returns the typed payload", async () => {
    fetchMock.mockResolvedValue(jsonResponse(200, HEALTH));
    await expect(getHealth()).resolves.toEqual(HEALTH);
    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(fetchMock.mock.calls[0][0]).toBe("/api/health");
  });
});

describe("uploadReport", () => {
  it("POSTs one multipart file field named report", async () => {
    fetchMock.mockResolvedValue(jsonResponse(201, UPLOAD));
    const file = new File(["{}"], "report.json", { type: "application/json" });
    await expect(uploadReport(file)).resolves.toEqual(UPLOAD);

    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe("/api/reports");
    expect(init?.method).toBe("POST");
    expect(init?.body).toBeInstanceOf(FormData);
    expect((init?.body as FormData).get("report")).toBe(file);
    // The browser must set the multipart boundary; the client must not override it.
    expect(init?.headers).toBeUndefined();
  });
});

describe("searchReports", () => {
  it("POSTs JSON to the search path", async () => {
    fetchMock.mockResolvedValue(jsonResponse(200, SEARCH));
    await expect(searchReports({ report_id: "r-1", query: "index" })).resolves.toEqual(SEARCH);

    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe("/api/search");
    expect(init?.method).toBe("POST");
    expect(init?.headers).toEqual({ "Content-Type": "application/json" });
    expect(JSON.parse(String(init?.body))).toEqual({ report_id: "r-1", query: "index" });
  });

  it("passes limit 0 through without client-side clamping", async () => {
    fetchMock.mockResolvedValue(jsonResponse(200, SEARCH));
    await searchReports({ report_id: "r-1", query: "index", limit: 0 });
    expect(JSON.parse(String(fetchMock.mock.calls[0][1]?.body)).limit).toBe(0);
  });
});

describe("error handling", () => {
  it("surfaces the uniform {code, detail} body as an ApiError", async () => {
    fetchMock.mockResolvedValue(
      jsonResponse(404, { code: "unknown_report", detail: "No report session with that id." }),
    );
    const error = await getHealth().catch((exc: unknown) => exc);
    expect(error).toBeInstanceOf(ApiError);
    const apiError = error as ApiError;
    expect(apiError.status).toBe(404);
    expect(apiError.code).toBe("unknown_report");
    expect(apiError.detail).toBe("No report session with that id.");
    expect(apiError.message).toBe(apiError.detail);
  });

  it("handles a non-JSON 500 without echoing the body", async () => {
    fetchMock.mockResolvedValue(
      new Response("Internal Server Error", {
        status: 500,
        headers: { "Content-Type": "text/plain" },
      }),
    );
    const error = (await getHealth().catch((exc: unknown) => exc)) as ApiError;
    expect(error).toBeInstanceOf(ApiError);
    expect(error.status).toBe(500);
    expect(error.code).toBe("http_500");
    expect(error.detail).not.toContain("Internal Server Error");
  });

  it("falls back when an error body is JSON but not the uniform shape", async () => {
    fetchMock.mockResolvedValue(jsonResponse(400, { detail: "framework-shaped" }));
    const error = (await getHealth().catch((exc: unknown) => exc)) as ApiError;
    expect(error).toBeInstanceOf(ApiError);
    expect(error.code).toBe("http_400");
  });

  it("reports a network failure as a client-side ApiError", async () => {
    fetchMock.mockRejectedValue(new TypeError("Failed to fetch"));
    const error = (await getHealth().catch((exc: unknown) => exc)) as ApiError;
    expect(error).toBeInstanceOf(ApiError);
    expect(error.status).toBe(0);
    expect(error.code).toBe("network_error");
  });

  it("refuses a successful response whose body is not valid JSON", async () => {
    fetchMock.mockResolvedValue(new Response("{not json", { status: 200 }));
    const error = (await getHealth().catch((exc: unknown) => exc)) as ApiError;
    expect(error).toBeInstanceOf(ApiError);
    expect(error.status).toBe(200);
    expect(error.code).toBe("invalid_response");
  });
});