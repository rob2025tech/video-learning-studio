import { describe, expect, it } from "vitest";
import { ApiError } from "./client";
import { describeApiError } from "./messages";

const SERVER_DETAIL = "raw-server-detail-that-must-never-render";

const CASES: ReadonlyArray<readonly [code: string, fragment: string]> = [
  ["invalid_report", "not a valid LearnLens report artifact"],
  ["invalid_encoding", "not a valid LearnLens report artifact"],
  ["invalid_multipart", "not a valid LearnLens report artifact"],
  ["missing_report", "not a valid LearnLens report artifact"],
  ["request_too_large", "upload limit"],
  ["unknown_report", "no longer exists"],
  ["report_expired", "expired"],
  ["network_error", "uv run video-learning-web"],
  ["forbidden_host", "host or origin"],
  ["forbidden_origin", "host or origin"],
  ["http_500", "unexpected response"],
  ["invalid_response", "unexpected response"],
];

describe("describeApiError", () => {
  it.each(CASES)("maps %s to an actionable message without echoing detail", (code, fragment) => {
    const message = describeApiError(new ApiError(400, code, SERVER_DETAIL));
    expect(message).toContain(fragment);
    expect(message).not.toContain(SERVER_DETAIL);
  });
});