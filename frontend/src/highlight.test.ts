import { describe, expect, it } from "vitest";
import type { MatchedSpan } from "./api/types";
import { highlightSegments } from "./highlight";

const TEXT = "the index ranks record 0";

function span(start: number, end: number, surface: string): MatchedSpan {
  return { term: surface.toLowerCase(), surface, start, end };
}

describe("highlightSegments", () => {
  it("returns plain text when there are no spans", () => {
    expect(highlightSegments(TEXT, [])).toEqual([{ text: TEXT, matched: false }]);
  });

  it("splits around a single match", () => {
    expect(highlightSegments(TEXT, [span(4, 9, "index")])).toEqual([
      { text: "the ", matched: false },
      { text: "index", matched: true },
      { text: " ranks record 0", matched: false },
    ]);
  });

  it("fuses nested spans into one outermost interval", () => {
    // "index" (4..9) contains a part "in" (4..6): both match, must not double-mark.
    expect(highlightSegments(TEXT, [span(4, 9, "index"), span(4, 6, "in")])).toEqual([
      { text: "the ", matched: false },
      { text: "index", matched: true },
      { text: " ranks record 0", matched: false },
    ]);
  });

  it("fuses overlapping spans", () => {
    expect(highlightSegments(TEXT, [span(4, 9, "index"), span(7, 13, "ex ran")])).toEqual([
      { text: "the ", matched: false },
      { text: "index ran", matched: true },
      { text: "ks record 0", matched: false },
    ]);
  });

  it("keeps disjoint matches separate and in order", () => {
    expect(highlightSegments(TEXT, [span(16, 22, "record"), span(4, 9, "index")])).toEqual([
      { text: "the ", matched: false },
      { text: "index", matched: true },
      { text: " ranks ", matched: false },
      { text: "record", matched: true },
      { text: " 0", matched: false },
    ]);
  });

  it("handles a match at the very start and end without empty segments", () => {
    expect(highlightSegments("index", [span(0, 5, "index")])).toEqual([
      { text: "index", matched: true },
    ]);
  });

  it("falls back to plain text when a span breaks the surface invariant", () => {
    expect(highlightSegments(TEXT, [span(4, 9, "WRONG")])).toEqual([
      { text: TEXT, matched: false },
    ]);
  });

  it("falls back to plain text when a span is out of range", () => {
    expect(highlightSegments(TEXT, [span(4, 999, "index")])).toEqual([
      { text: TEXT, matched: false },
    ]);
  });

  it("preserves the verbatim text across all segments", () => {
    const segments = highlightSegments(TEXT, [span(4, 9, "index"), span(16, 22, "record")]);
    expect(segments.map((segment) => segment.text).join("")).toBe(TEXT);
  });
});