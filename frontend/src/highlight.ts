import type { MatchedSpan } from "./api/types";

export interface HighlightSegment {
  text: string;
  matched: boolean;
}

/**
 * Split verbatim evidence text into ordered plain/matched segments for rendering.
 *
 * Spans may nest (a compound and its parts can both match), so they are fused into
 * outermost intervals first. Every offset is validated against the core's
 * `text[start:end] === surface` invariant; if any span breaks it, the whole text is
 * returned unhighlighted rather than slicing on a bad offset — the same fallback the
 * CLI renderer uses. The matched surface is always the raw slice, never rewritten.
 */
export function highlightSegments(
  text: string,
  spans: readonly MatchedSpan[],
): HighlightSegment[] {
  const valid =
    spans.length > 0 &&
    spans.every(
      (span) =>
        Number.isInteger(span.start) &&
        Number.isInteger(span.end) &&
        span.start >= 0 &&
        span.start < span.end &&
        span.end <= text.length &&
        text.slice(span.start, span.end) === span.surface,
    );
  if (!valid) {
    return [{ text, matched: false }];
  }

  const sorted = [...spans].sort((a, b) => a.start - b.start || a.end - b.end);
  const intervals: Array<[number, number]> = [];
  for (const span of sorted) {
    const last = intervals[intervals.length - 1];
    if (last && span.start <= last[1]) {
      last[1] = Math.max(last[1], span.end);
    } else {
      intervals.push([span.start, span.end]);
    }
  }

  const segments: HighlightSegment[] = [];
  let cursor = 0;
  for (const [start, end] of intervals) {
    if (start > cursor) {
      segments.push({ text: text.slice(cursor, start), matched: false });
    }
    segments.push({ text: text.slice(start, end), matched: true });
    cursor = end;
  }
  if (cursor < text.length) {
    segments.push({ text: text.slice(cursor), matched: false });
  }
  return segments;
}