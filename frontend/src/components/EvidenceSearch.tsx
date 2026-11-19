import { useEffect, useRef, useState } from "react";
import { ApiError, searchReports } from "../api/client";
import { describeApiError } from "../api/messages";
import type { SearchMatch, SearchResponse, UploadResponse } from "../api/types";
import { formatTimestamp } from "../format";
import { highlightSegments } from "../highlight";

const LIMIT_OPTIONS = [5, 10, 20, 50] as const;
const MAX_QUERY_LENGTH = 512;
const GENERIC_FAILURE = "The search failed for an unexpected reason. Try again.";

type SearchPhase = "idle" | "searching" | "results" | "error";

interface EvidenceSearchProps {
  session: UploadResponse;
  onNavigateUpload: () => void;
  onSeek?: (seconds: number, evidenceId: string) => void;
  selectedEvidenceId?: string | null;
  hasVideo?: boolean;
}

export default function EvidenceSearch({
  session,
  onNavigateUpload,
  onSeek,
  selectedEvidenceId = null,
  hasVideo = false,
}: EvidenceSearchProps) {
  const [query, setQuery] = useState("");
  const [limit, setLimit] = useState<number>(LIMIT_OPTIONS[0]);
  const [phase, setPhase] = useState<SearchPhase>("idle");
  const [response, setResponse] = useState<SearchResponse | null>(null);
  const [error, setError] = useState<ApiError | null>(null);
  // Monotonic request identity: a response whose sequence is no longer current
  // (superseded query, or a report change) must not land in state.
  const searchSeq = useRef(0);

  // A new report invalidates any results held against the previous session.
  useEffect(() => {
    searchSeq.current += 1;
    setQuery("");
    setPhase("idle");
    setResponse(null);
    setError(null);
  }, [session.report_id]);

  const trimmed = query.trim();
  const searching = phase === "searching";
  const canSearch = trimmed.length > 0 && !searching;

  const runSearch = async () => {
    if (!canSearch) {
      return;
    }
    const seq = ++searchSeq.current;
    setPhase("searching");
    setError(null);
    try {
      const result = await searchReports({
        report_id: session.report_id,
        query: trimmed,
        limit,
      });
      if (seq !== searchSeq.current) {
        return;
      }
      setResponse(result);
      setPhase("results");
    } catch (exc) {
      if (seq !== searchSeq.current) {
        return;
      }
      setError(exc instanceof ApiError ? exc : null);
      setPhase("error");
    }
  };

  const sessionLost = error?.code === "unknown_report" || error?.code === "report_expired";

  return (
    <section className="card search-panel" aria-labelledby="search-heading">
      <div className="search-head">
        <h2 id="search-heading">Evidence search</h2>
        <p className="active-report">
          Searching <strong>{session.filename}</strong>
        </p>
      </div>

      <form
        className="search-form"
        onSubmit={(event) => {
          event.preventDefault();
          void runSearch();
        }}
      >
        <div className="search-field">
          <label htmlFor="search-query">Search query</label>
          <input
            id="search-query"
            type="text"
            value={query}
            maxLength={MAX_QUERY_LENGTH}
            placeholder="e.g. search index"
            autoComplete="off"
            aria-describedby="search-query-hint"
            disabled={searching}
            onChange={(event) => setQuery(event.target.value)}
          />
        </div>
        <div className="search-field search-limit">
          <label htmlFor="search-limit">Results limit</label>
          <select
            id="search-limit"
            value={limit}
            disabled={searching}
            onChange={(event) => setLimit(Number(event.target.value))}
          >
            {LIMIT_OPTIONS.map((option) => (
              <option key={option} value={option}>
                {option}
              </option>
            ))}
          </select>
        </div>
        <button type="submit" className="button" disabled={!canSearch}>
          {searching ? "Searching…" : "Search"}
        </button>
      </form>
      <p id="search-query-hint" className="search-hint">
        {query.length > 0 && trimmed.length === 0
          ? "Enter a query with at least one non-space character."
          : "Matches spoken words and on-screen text. The limit caps what is returned, never what is considered."}
      </p>

      {searching && (
        <p role="status" className="status-text">
          Searching {session.filename}…
        </p>
      )}

      {phase === "error" && error && (
        <div role="alert" className="search-error">
          <p className="error-text">{describeApiError(error)}</p>
          {sessionLost && (
            <button type="button" className="button" onClick={onNavigateUpload}>
              Go to report upload
            </button>
          )}
        </div>
      )}
      {phase === "error" && !error && <p role="alert" className="error-text">{GENERIC_FAILURE}</p>}

      {phase === "results" && response && (
        <SearchResultsView
          response={response}
          hasVideo={hasVideo}
          selectedEvidenceId={selectedEvidenceId}
          onActivate={(match) => onSeek?.(match.start_seconds, match.evidence_id)}
        />
      )}
    </section>
  );
}

function SearchResultsView({
  response,
  hasVideo,
  selectedEvidenceId,
  onActivate,
}: {
  response: SearchResponse;
  hasVideo: boolean;
  selectedEvidenceId: string | null;
  onActivate: (match: SearchMatch) => void;
}) {
  const [noVideoNotice, setNoVideoNotice] = useState(false);
  const summaryRef = useRef<HTMLParagraphElement>(null);
  const coverage = response.coverage;
  const capped = coverage.returned_record_count < coverage.matched_record_count;

  // Move focus to the results summary on each completed search so keyboard and
  // screen-reader users land on the answer, not back on the Search button. Skip
  // when the search view is hidden (a background completion must not steal focus).
  useEffect(() => {
    const node = summaryRef.current;
    if (node && !node.closest("[hidden]")) {
      node.focus();
    }
  }, [response]);

  // Selecting a video resolves the "select a video" prompt; drop any stale notice.
  useEffect(() => {
    if (hasVideo) {
      setNoVideoNotice(false);
    }
  }, [hasVideo]);

  const activate = (match: SearchMatch) => {
    setNoVideoNotice(!hasVideo);
    onActivate(match);
  };

  return (
    <div className="search-results">
      <p className="results-summary" role="status" ref={summaryRef} tabIndex={-1}>
        <span className={`status-pill status-${response.status}`}>{statusLabel(response.status)}</span>
        {coverage.matched_record_count} matching record
        {coverage.matched_record_count === 1 ? "" : "s"} of {coverage.searchable_record_count}{" "}
        searchable.
      </p>

      {!hasVideo && (
        <p className="video-needed-hint">
          Timestamp navigation needs a local video — select one in the Local video panel above.
          Search itself works without it.
        </p>
      )}
      {noVideoNotice && (
        <p role="status" className="video-needed-hint">
          Select a local video to jump to this timestamp.
        </p>
      )}

      {response.results.length === 0 ? (
        <p className="empty-explanation">{emptyExplanation(response)}</p>
      ) : (
        <ol className="result-list">
          {response.results.map((match) => (
            <ResultCard
              key={match.evidence_id}
              match={match}
              selected={match.evidence_id === selectedEvidenceId}
              hasVideo={hasVideo}
              onActivate={activate}
            />
          ))}
        </ol>
      )}

      {capped && (
        <p className="capped-note">
          Showing {coverage.returned_record_count} of {coverage.matched_record_count} matching
          records (limit {response.limit}). Raise the limit to see more.
        </p>
      )}
    </div>
  );
}

function ResultCard({
  match,
  selected,
  hasVideo,
  onActivate,
}: {
  match: SearchMatch;
  selected: boolean;
  hasVideo: boolean;
  onActivate: (match: SearchMatch) => void;
}) {
  const segments = highlightSegments(match.text, match.matched_spans);
  const isSpeech = match.source_type === "speech";
  const attribution =
    match.segment_id === null ? "unattributed" : `segment ${match.segment_id}`;
  const timestamp =
    isSpeech && match.end_seconds !== null
      ? `${formatTimestamp(match.start_seconds)} → ${formatTimestamp(match.end_seconds)}`
      : formatTimestamp(match.start_seconds);

  return (
    <li className={`result-card ${isSpeech ? "is-speech" : "is-ocr"}${selected ? " is-selected" : ""}`}>
      <button
        type="button"
        className="result-activate"
        aria-current={selected ? "true" : undefined}
        aria-label={`Jump to ${timestamp}, ${isSpeech ? "speech" : "OCR"} evidence`}
        onClick={() => onActivate(match)}
      >
        <span className="result-meta">
          <span className={`source-badge ${isSpeech ? "badge-speech" : "badge-ocr"}`}>
            {isSpeech ? "SPEECH" : "OCR"}
          </span>
          <span className="timestamp">{timestamp}</span>
          <span className="result-attribution">{attribution}</span>
          <span className="result-score">BM25 {match.score.toFixed(3)}</span>
        </span>
        <span className="result-text">
          {segments.map((segment, index) =>
            segment.matched ? (
              <mark key={index}>{segment.text}</mark>
            ) : (
              <span key={index}>{segment.text}</span>
            ),
          )}
        </span>
        {match.matched_terms.length > 0 && (
          <span className="matched-terms">matched {match.matched_terms.join(", ")}</span>
        )}
        <span className="result-jump-hint">
          {selected ? "Selected" : hasVideo ? "Jump to timestamp" : "Select a video to jump"}
        </span>
      </button>
    </li>
  );
}

function statusLabel(status: SearchResponse["status"]): string {
  switch (status) {
    case "ok":
      return "ok";
    case "no_confident_match":
      return "no confident match";
    case "no_evidence":
      return "no evidence";
  }
}

// Port of cli/render.py _explain_empty_answer: the four empty shapes are distinct
// user problems, named from the core's coverage counts rather than a generic message.
function emptyExplanation(response: SearchResponse): string {
  const coverage = response.coverage;
  if (response.status === "no_evidence") {
    return `Nothing in this artifact is searchable: ${coverage.record_count} records projected, ${coverage.searchable_record_count} of them carrying a meaningful term.`;
  }
  if (coverage.matched_record_count > 0) {
    return `${coverage.matched_record_count} records matched, but the limit (${response.limit}) asked for none.`;
  }
  if (coverage.meaningful_query_term_count === 0) {
    return `The query split into ${coverage.query_token_count} fragments, none of them a meaningful term (stopwords or UI chrome), so nothing was scored.`;
  }
  return `None of the ${coverage.meaningful_query_term_count} meaningful query terms appears in the ${coverage.searchable_record_count} searchable records.`;
}