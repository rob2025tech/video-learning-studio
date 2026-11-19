import { act, cleanup, fireEvent, render, screen } from "@testing-library/react";
import { createRef, useState, type RefObject } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import VideoPlayer, { type VideoPlayerHandle } from "./VideoPlayer";

const createObjectURL = vi.fn();
const revokeObjectURL = vi.fn();
const scrollIntoView = vi.fn();
let urlCounter = 0;

// jsdom ships no matchMedia; the seek path queries prefers-reduced-motion through it.
function mockMatchMedia(prefersReducedMotion: boolean): void {
  Object.defineProperty(window, "matchMedia", {
    writable: true,
    configurable: true,
    value: vi.fn().mockImplementation((query: string) => ({
      matches: prefersReducedMotion,
      media: query,
      onchange: null,
      addListener: vi.fn(),
      removeListener: vi.fn(),
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
      dispatchEvent: vi.fn(),
    })),
  });
}

function makeFile(name = "clip.mp4", type = "video/mp4"): File {
  return new File(["synthetic-bytes"], name, { type });
}

function Harness({
  reportDurationSeconds = 24,
  playerRef,
}: {
  reportDurationSeconds?: number | null;
  playerRef?: RefObject<VideoPlayerHandle | null>;
}) {
  const [file, setFile] = useState<File | null>(null);
  return (
    <VideoPlayer
      ref={playerRef}
      file={file}
      reportDurationSeconds={reportDurationSeconds}
      onSelect={setFile}
      onClear={() => setFile(null)}
    />
  );
}

function selectVideo(name = "clip.mp4", type = "video/mp4"): File {
  const file = makeFile(name, type);
  fireEvent.change(screen.getByLabelText("Original video (local file)"), {
    target: { files: [file] },
  });
  return file;
}

function videoElement(container: HTMLElement): HTMLVideoElement {
  const video = container.querySelector("video");
  if (!video) {
    throw new Error("expected a video element");
  }
  return video;
}

function setDuration(video: HTMLVideoElement, value: number): void {
  Object.defineProperty(video, "duration", { value, configurable: true });
}

beforeEach(() => {
  urlCounter = 0;
  createObjectURL.mockReset();
  revokeObjectURL.mockReset();
  createObjectURL.mockImplementation((file: File) => `blob:mock-${file.name}-${++urlCounter}`);
  Object.defineProperty(URL, "createObjectURL", {
    value: createObjectURL,
    configurable: true,
    writable: true,
  });
  Object.defineProperty(URL, "revokeObjectURL", {
    value: revokeObjectURL,
    configurable: true,
    writable: true,
  });
  vi.spyOn(HTMLMediaElement.prototype, "play").mockImplementation(() => Promise.resolve());
  vi.spyOn(HTMLMediaElement.prototype, "pause").mockImplementation(() => {});
  vi.spyOn(HTMLMediaElement.prototype, "canPlayType").mockReturnValue("probably");
  scrollIntoView.mockReset();
  Element.prototype.scrollIntoView = scrollIntoView;
  mockMatchMedia(false);
});

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

describe("VideoPlayer selection and object URL lifecycle", () => {
  it("shows the missing-video state before any selection", () => {
    render(<Harness />);
    expect(screen.getByText("No video selected.")).toBeTruthy();
    expect(createObjectURL).not.toHaveBeenCalled();
  });

  it("creates an object URL and shows the filename with an unverified label", () => {
    render(<Harness />);
    selectVideo("tutorial.mp4");
    expect(createObjectURL).toHaveBeenCalledTimes(1);
    expect(screen.getByText("tutorial.mp4")).toBeTruthy();
    expect(screen.getByText(/not verified against the report/)).toBeTruthy();
    expect(screen.getByLabelText("Local video player")).toBeTruthy();
  });

  it("never sends the selected file to a backend endpoint", () => {
    const fetchSpy = vi.spyOn(globalThis, "fetch").mockImplementation(() =>
      Promise.resolve(new Response("{}")),
    );
    render(<Harness />);
    selectVideo();
    expect(fetchSpy).not.toHaveBeenCalled();
  });

  it("revokes the previous URL when replacing a video", () => {
    render(<Harness />);
    selectVideo("first.mp4");
    const firstUrl = createObjectURL.mock.results[0].value as string;
    selectVideo("second.mp4");
    expect(revokeObjectURL).toHaveBeenCalledWith(firstUrl);
    expect(createObjectURL).toHaveBeenCalledTimes(2);
    expect(screen.getByText("second.mp4")).toBeTruthy();
  });

  it("revokes the active URL on unmount", () => {
    const { unmount } = render(<Harness />);
    selectVideo("clip.mp4");
    const url = createObjectURL.mock.results[0].value as string;
    unmount();
    expect(revokeObjectURL).toHaveBeenCalledWith(url);
  });

  it("does not leak URLs across repeated selection and unmount", () => {
    const { unmount } = render(<Harness />);
    selectVideo("a.mp4");
    selectVideo("b.mp4");
    selectVideo("c.mp4");
    unmount();
    expect(createObjectURL).toHaveBeenCalledTimes(3);
    expect(revokeObjectURL).toHaveBeenCalledTimes(3);
  });

  it("clears back to the missing-video state and revokes on remove", () => {
    render(<Harness />);
    selectVideo("clip.mp4");
    const url = createObjectURL.mock.results[0].value as string;
    fireEvent.click(screen.getByRole("button", { name: "Remove video" }));
    expect(revokeObjectURL).toHaveBeenCalledWith(url);
    expect(screen.getByText("No video selected.")).toBeTruthy();
  });
});

describe("VideoPlayer playback states", () => {
  it("shows a metadata-loading state until metadata arrives", () => {
    const { container } = render(
      <VideoPlayer file={makeFile()} reportDurationSeconds={24} onSelect={vi.fn()} onClear={vi.fn()} />,
    );
    expect(screen.getByText("Loading video metadata…")).toBeTruthy();
    const video = videoElement(container);
    setDuration(video, 24);
    fireEvent(video, new Event("loadedmetadata"));
    expect(screen.queryByText("Loading video metadata…")).toBeNull();
  });

  it("reaches a ready state on canplay", () => {
    const { container } = render(
      <VideoPlayer file={makeFile()} reportDurationSeconds={24} onSelect={vi.fn()} onClear={vi.fn()} />,
    );
    fireEvent(videoElement(container), new Event("canplay"));
    expect(screen.queryByText("Loading video metadata…")).toBeNull();
    expect(screen.queryByRole("alert")).toBeNull();
  });

  it("reports unsupported media from the error event, not from canPlayType", () => {
    const { container } = render(
      <VideoPlayer file={makeFile()} reportDurationSeconds={24} onSelect={vi.fn()} onClear={vi.fn()} />,
    );
    // canPlayType says "probably" yet the real error event must still win.
    fireEvent(videoElement(container), new Event("error"));
    const alert = screen.getByRole("alert");
    expect(alert.textContent).toContain("could not be played");
  });

  it("replaces cleanly while metadata is still loading", () => {
    const { container } = render(<Harness />);
    selectVideo("slow.mp4");
    const firstUrl = createObjectURL.mock.results[0].value as string;
    expect(screen.getByText("Loading video metadata…")).toBeTruthy();
    selectVideo("fast.mp4");
    expect(revokeObjectURL).toHaveBeenCalledWith(firstUrl);
    // The old element is gone (keyed by URL); the new one is still loading.
    expect(videoElement(container).getAttribute("src")).not.toBe(firstUrl);
    expect(screen.getByText("Loading video metadata…")).toBeTruthy();
  });
});

describe("VideoPlayer duration consistency warning", () => {
  function loadWithDurations(report: number | null, video: number) {
    const { container } = render(
      <VideoPlayer
        file={makeFile()}
        reportDurationSeconds={report}
        onSelect={vi.fn()}
        onClear={vi.fn()}
      />,
    );
    const element = videoElement(container);
    setDuration(element, video);
    fireEvent(element, new Event("loadedmetadata"));
  }

  it("stays silent when durations match", () => {
    loadWithDurations(24, 24.4);
    expect(screen.queryByText(/differs from the report/)).toBeNull();
  });

  it("warns when the difference exceeds one second", () => {
    loadWithDurations(24, 30);
    expect(screen.getByText(/differs from the report/)).toBeTruthy();
  });

  it("skips the comparison when the report duration is unknown", () => {
    loadWithDurations(null, 30);
    expect(screen.queryByText(/differs from the report/)).toBeNull();
  });

  it("skips the comparison when the video duration is unknown", () => {
    loadWithDurations(24, 0);
    expect(screen.queryByText(/differs from the report/)).toBeNull();
  });

  it("skips the comparison for an invalid video duration", () => {
    loadWithDurations(24, Number.POSITIVE_INFINITY);
    expect(screen.queryByText(/differs from the report/)).toBeNull();
  });

  it("does not block playback when it warns", () => {
    loadWithDurations(24, 30);
    expect(screen.getByLabelText("Local video player")).toBeTruthy();
    expect(screen.queryByRole("alert")).toBeNull();
  });
});

describe("VideoPlayer seeking", () => {
  it("seeks to the requested offset without playing", () => {
    const ref = createRef<VideoPlayerHandle>();
    const playSpy = vi.spyOn(HTMLMediaElement.prototype, "play");
    const { container } = render(
      <VideoPlayer
        ref={ref}
        file={makeFile()}
        reportDurationSeconds={24}
        onSelect={vi.fn()}
        onClear={vi.fn()}
      />,
    );
    const video = videoElement(container);
    act(() => {
      expect(ref.current?.seek(12.5)).toBe(true);
    });
    expect(video.currentTime).toBe(12.5);
    expect(playSpy).not.toHaveBeenCalled();
    expect(screen.getByText("Jumped to 00:00:12.5")).toBeTruthy();
  });

  it("scrolls the player into view on seek", () => {
    const ref = createRef<VideoPlayerHandle>();
    render(
      <VideoPlayer
        ref={ref}
        file={makeFile()}
        reportDurationSeconds={24}
        onSelect={vi.fn()}
        onClear={vi.fn()}
      />,
    );
    ref.current?.seek(3);
    expect(scrollIntoView).toHaveBeenCalled();
  });

  it("returns false rather than throwing when no video is selected", () => {
    const ref = createRef<VideoPlayerHandle>();
    render(<Harness playerRef={ref} />);
    expect(ref.current?.seek(5)).toBe(false);
  });
});

describe("VideoPlayer seek honesty and motion preference", () => {
  function renderReadyPlayer() {
    const ref = createRef<VideoPlayerHandle>();
    const { container } = render(
      <VideoPlayer
        ref={ref}
        file={makeFile()}
        reportDurationSeconds={24}
        onSelect={vi.fn()}
        onClear={vi.fn()}
      />,
    );
    const video = videoElement(container);
    setDuration(video, 24);
    fireEvent(video, new Event("loadedmetadata"));
    fireEvent(video, new Event("canplay"));
    return { ref, video };
  }

  it("does not claim a successful seek while in the error phase", () => {
    const ref = createRef<VideoPlayerHandle>();
    const { container } = render(
      <VideoPlayer
        ref={ref}
        file={makeFile()}
        reportDurationSeconds={24}
        onSelect={vi.fn()}
        onClear={vi.fn()}
      />,
    );
    const video = videoElement(container);
    fireEvent(video, new Event("error"));
    expect(screen.getByRole("alert")).toBeTruthy();

    let sought: boolean | undefined;
    act(() => {
      sought = ref.current?.seek(12);
    });
    expect(sought).toBe(false);
    expect(video.currentTime).toBe(0);
    expect(screen.queryByText(/Jumped to/)).toBeNull();
  });

  it("scrolls smoothly when motion is allowed", () => {
    const { ref } = renderReadyPlayer();
    act(() => {
      expect(ref.current?.seek(6)).toBe(true);
    });
    expect(scrollIntoView).toHaveBeenCalledWith({ behavior: "smooth", block: "nearest" });
  });

  it("avoids smooth scrolling when reduced motion is preferred", () => {
    mockMatchMedia(true);
    const { ref } = renderReadyPlayer();
    act(() => {
      expect(ref.current?.seek(6)).toBe(true);
    });
    expect(scrollIntoView).toHaveBeenCalledWith({ behavior: "auto", block: "nearest" });
    expect(scrollIntoView).not.toHaveBeenCalledWith({ behavior: "smooth", block: "nearest" });
  });
});