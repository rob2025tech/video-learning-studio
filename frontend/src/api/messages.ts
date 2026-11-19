import type { ApiError } from "./client";

/**
 * User-facing text per stable API error code. Classification is by `code` only;
 * the server's `detail` is never shown, and no raw response body is echoed.
 * `unknown_report` / `report_expired` come from the search route, not upload —
 * they are mapped here so the search milestone reuses one vocabulary.
 */
export function describeApiError(error: ApiError): string {
  switch (error.code) {
    case "invalid_report":
    case "invalid_encoding":
    case "invalid_multipart":
    case "missing_report":
      return "This file is not a valid LearnLens report artifact. Choose a report JSON created by the video-learning report command — not a video file.";
    case "request_too_large":
      return "This report is larger than the API's configured upload limit. Choose a smaller report artifact.";
    case "unknown_report":
      return "That report session no longer exists. Upload the report again to continue.";
    case "report_expired":
      return "That report session expired. Upload the report again to continue.";
    case "network_error":
      return "Could not reach the local API. Start it with: uv run video-learning-web";
    case "forbidden_host":
    case "forbidden_origin":
      return "The API refused this request's host or origin. Open the app through the Vite dev server (npm run dev) so requests go through the local proxy.";
    default:
      return "The API returned an unexpected response. Try again; if it keeps happening, restart the local API.";
  }
}