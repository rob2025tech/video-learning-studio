import type {
  ApiErrorBody,
  HealthResponse,
  SearchRequest,
  SearchResponse,
  UploadResponse,
} from "./types";

/** Relative by design: the Vite dev proxy forwards /api to the loopback FastAPI server. */
export const API_BASE = "/api";

/**
 * One error type for every failure mode. `code` is the backend's stable
 * machine-readable identifier (branch on it, never on `detail`); for a non-JSON
 * error body — an unexpected 500 is text/plain — the client synthesizes the same
 * `http_<status>` fallback the backend uses for unmapped statuses, and never
 * echoes the raw body.
 */
export class ApiError extends Error {
  readonly status: number;
  readonly code: string;
  readonly detail: string;

  constructor(status: number, code: string, detail: string) {
    super(detail);
    this.name = "ApiError";
    this.status = status;
    this.code = code;
    this.detail = detail;
  }
}

function isApiErrorBody(value: unknown): value is ApiErrorBody {
  if (typeof value !== "object" || value === null) {
    return false;
  }
  const record = value as Record<string, unknown>;
  return typeof record.code === "string" && typeof record.detail === "string";
}

async function toApiError(response: Response): Promise<ApiError> {
  let body: unknown;
  try {
    body = await response.json();
  } catch {
    body = undefined;
  }
  if (isApiErrorBody(body)) {
    return new ApiError(response.status, body.code, body.detail);
  }
  return new ApiError(
    response.status,
    `http_${response.status}`,
    "The API returned an unexpected error response.",
  );
}

async function requestJson<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE}${path}`, init);
  } catch {
    throw new ApiError(0, "network_error", "Could not reach the local LearnLens API.");
  }
  if (!response.ok) {
    throw await toApiError(response);
  }
  try {
    return (await response.json()) as T;
  } catch {
    throw new ApiError(response.status, "invalid_response", "The API response was not valid JSON.");
  }
}

export function getHealth(): Promise<HealthResponse> {
  return requestJson<HealthResponse>("/health");
}

export function uploadReport(file: File): Promise<UploadResponse> {
  const form = new FormData();
  form.append("report", file);
  // No Content-Type header: the browser must set the multipart boundary itself.
  return requestJson<UploadResponse>("/reports", { method: "POST", body: form });
}

export function searchReports(request: SearchRequest): Promise<SearchResponse> {
  return requestJson<SearchResponse>("/search", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(request),
  });
}