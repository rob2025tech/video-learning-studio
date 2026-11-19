import { describe, expect, it } from "vitest";
import { formatBytes, formatDuration, formatTimestamp } from "./format";

describe("formatTimestamp", () => {
  it("formats a whole offset as HH:MM:SS.s", () => {
    expect(formatTimestamp(0)).toBe("00:00:00.0");
    expect(formatTimestamp(3.8)).toBe("00:00:03.8");
    expect(formatTimestamp(62.5)).toBe("00:01:02.5");
    expect(formatTimestamp(3661.2)).toBe("01:01:01.2");
  });

  it("carries a rounding tenth into the seconds field", () => {
    expect(formatTimestamp(0.96)).toBe("00:00:01.0");
    expect(formatTimestamp(59.97)).toBe("00:01:00.0");
  });

  it("clamps negative offsets to zero", () => {
    expect(formatTimestamp(-5)).toBe("00:00:00.0");
  });
});

describe("formatBytes", () => {
  it("scales units", () => {
    expect(formatBytes(512)).toBe("512 B");
    expect(formatBytes(3266)).toBe("3.2 KB");
    expect(formatBytes(2 * 1024 * 1024)).toBe("2.0 MB");
  });
});

describe("formatDuration", () => {
  it("scales units", () => {
    expect(formatDuration(24)).toBe("24 s");
    expect(formatDuration(1800)).toBe("30 min");
    expect(formatDuration(3600)).toBe("1 h");
    expect(formatDuration(3900)).toBe("1 h 5 min");
  });
});