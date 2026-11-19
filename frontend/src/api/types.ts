/**
 * Wire types for the LearnLens local evidence API (M1).
 *
 * Every field here mirrors a verified backend source:
 * - health: api/app.py `health()`
 * - upload: api/reports.py `upload_payload()`
 * - search: core/search.py `SearchResults.to_dict()` plus the two provenance
 *   changes in api/responses.py `search_payload()` (basename `source`,
 *   `source_is_redacted: true`)
 * - errors: api/errors.py `ApiError.to_dict()` — exactly {code, detail}
 *
 * The backend types `transcription_status` as a plain str (known values:
 * "ok", "no-audio", "unavailable"), so it stays `string` here rather than a
 * union the artifact could violate.
 */

export interface HealthResponse {
  status: "ok";
  version: string;
  report_schema: string;
  search_schema: string;
  active_report_sessions: number;
  max_report_sessions: number;
  max_report_bytes: number;
}

export interface CorpusCounts {
  record_count: number;
  searchable_record_count: number;
  speech_record_count: number;
  ocr_record_count: number;
  segment_count: number;
}

export interface UploadResponse {
  report_id: string;
  report_schema: string;
  filename: string;
  bytes_received: number;
  expires_in_seconds: number;
  video_duration_seconds: number | null;
  transcription_status: string;
  transcription_available: boolean;
  corpus: CorpusCounts;
}

export interface SearchRequest {
  report_id: string;
  query?: string;
  /** Deliberately unconstrained: 0 or negative is legal and returns full coverage with no records. */
  limit?: number;
}

export type SearchSourceType = "speech" | "ocr";

export type SearchStatus = "ok" | "no_confident_match" | "no_evidence";

export interface MatchedSpan {
  term: string;
  surface: string;
  start: number;
  end: number;
}

export interface SearchMatch {
  evidence_id: string;
  source_type: SearchSourceType;
  text: string;
  start_seconds: number;
  end_seconds: number | null;
  segment_id: number | null;
  score: number;
  matched_terms: string[];
  matched_spans: MatchedSpan[];
}

export interface SearchCoverage {
  record_count: number;
  searchable_record_count: number;
  speech_record_count: number;
  ocr_record_count: number;
  attributed_record_count: number;
  unattributed_record_count: number;
  query_token_count: number;
  meaningful_query_term_count: number;
  matched_record_count: number;
  returned_record_count: number;
}

export interface SearchResponse {
  schema: string;
  query: string;
  status: SearchStatus;
  source: string;
  source_is_redacted: boolean;
  study_evidence_schema: string;
  transcription_status: string;
  transcription_available: boolean;
  video_duration_seconds: number | null;
  limit: number;
  results: SearchMatch[];
  result_count: number;
  coverage: SearchCoverage;
}

export interface ApiErrorBody {
  code: string;
  detail: string;
}