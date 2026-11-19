import {
  forwardRef,
  useEffect,
  useImperativeHandle,
  useRef,
  useState,
  type ChangeEvent,
} from "react";
import { formatTimestamp } from "../format";

export interface VideoPlayerHandle {
  /** Seek the local player to a media offset. Returns false when no video is loaded. */
  seek: (seconds: number) => boolean;
}

interface VideoPlayerProps {
  /** Owned by the app and keyed to the active report_id; null when no video is selected. */
  file: File | null;
  /** The report's authoritative duration, used only for a non-blocking consistency hint. */
  reportDurationSeconds: number | null;
  onSelect: (file: File) => void;
  onClear: () => void;
  /** False while the search view is hidden; pauses playback but keeps position. */
  active?: boolean;
}

type PlayerPhase = "empty" | "loading" | "ready" | "error";

const DURATION_TOLERANCE_SECONDS = 1;
const PLAYBACK_ERROR =
  "This video could not be played. Your browser may not support its format, or the file may be damaged. Choose another video.";

function isValidDuration(value: number | null | undefined): value is number {
  return typeof value === "number" && Number.isFinite(value) && value > 0;
}

/**
 * A browser-local player. The selected file never leaves the machine: it is played
 * through an object URL and is never sent to fetch, uploadReport, or any endpoint.
 * The object URL is created in one effect keyed on the file and revoked in that
 * effect's cleanup, which covers both replacement and unmount without leaking.
 */
const VideoPlayer = forwardRef<VideoPlayerHandle, VideoPlayerProps>(function VideoPlayer(
  { file, reportDurationSeconds, onSelect, onClear, active = true },
  ref,
) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const frameRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const [objectUrl, setObjectUrl] = useState<string | null>(null);
  const [phase, setPhase] = useState<PlayerPhase>("empty");
  const [buffering, setBuffering] = useState(false);
  const [videoDuration, setVideoDuration] = useState<number | null>(null);
  const [durationMismatch, setDurationMismatch] = useState(false);
  const [formatHint, setFormatHint] = useState(false);
  const [seekedTo, setSeekedTo] = useState<number | null>(null);

  useEffect(() => {
    setSeekedTo(null);
    setBuffering(false);
    setVideoDuration(null);
    setDurationMismatch(false);
    if (!file) {
      setObjectUrl(null);
      setPhase("empty");
      setFormatHint(false);
      return;
    }
    const url = URL.createObjectURL(file);
    setObjectUrl(url);
    setPhase("loading");
    // Preliminary only: the media error event is the authoritative failure signal.
    const probe = document.createElement("video");
    setFormatHint(file.type !== "" && probe.canPlayType(file.type) === "");
    return () => {
      URL.revokeObjectURL(url);
    };
  }, [file]);

  // Hidden view: pause so audio does not continue off-screen, but keep currentTime
  // so returning to the search view resumes at the same position.
  useEffect(() => {
    if (!active) {
      videoRef.current?.pause();
    }
  }, [active]);

  useImperativeHandle(
    ref,
    () => ({
      seek(seconds: number) {
        const video = videoRef.current;
        // An unplayable (error-phase) video cannot honour a seek; report failure
        // rather than claiming a jump that will never happen.
        if (!video || !objectUrl || phase === "error") {
          return false;
        }
        video.currentTime = seconds;
        setSeekedTo(seconds);
        const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
        frameRef.current?.scrollIntoView({
          behavior: reduceMotion ? "auto" : "smooth",
          block: "nearest",
        });
        return true;
      },
    }),
    [objectUrl, phase],
  );

  const handleLoadedMetadata = (event: ChangeEvent<HTMLVideoElement>) => {
    const duration = event.currentTarget.duration;
    const valid = isValidDuration(duration);
    setVideoDuration(valid ? duration : null);
    setPhase((current) => (current === "error" ? current : "ready"));
    if (valid && isValidDuration(reportDurationSeconds)) {
      setDurationMismatch(Math.abs(duration - reportDurationSeconds) > DURATION_TOLERANCE_SECONDS);
    } else {
      setDurationMismatch(false);
    }
  };

  const handleSelect = (event: ChangeEvent<HTMLInputElement>) => {
    const selected = event.target.files?.[0];
    if (selected) {
      onSelect(selected);
    }
    // Allow re-selecting the same file after a clear or an error.
    event.target.value = "";
  };

  const hasVideo = objectUrl !== null;

  return (
    <section className="card video-panel" aria-labelledby="video-heading">
      <div className="video-head">
        <h2 id="video-heading">Local video</h2>
        <p className="video-provenance">
          {file ? (
            <>
              <strong>{file.name}</strong>
              <span className="video-unverified">User-selected · not verified against the report</span>
            </>
          ) : (
            "Play the original video alongside its evidence. It stays on your machine."
          )}
        </p>
      </div>

      <div className="video-controls">
        <div className="video-field">
          <label htmlFor="local-video-file">Original video (local file)</label>
          <input
            ref={inputRef}
            id="local-video-file"
            type="file"
            accept="video/*"
            aria-describedby="video-select-hint"
            onChange={handleSelect}
          />
        </div>
        {file && (
          <button type="button" className="button button-quiet" onClick={onClear}>
            Remove video
          </button>
        )}
      </div>
      <p id="video-select-hint" className="video-hint">
        The video is played locally through a browser object URL and is never uploaded. Selecting a
        file does not prove it matches the report.
      </p>

      {durationMismatch && (
        <p role="status" className="video-warning">
          This video’s duration ({videoDuration === null ? "unknown" : formatTimestamp(videoDuration)})
          differs from the report ({formatTimestamp(reportDurationSeconds ?? 0)}). It may be the wrong
          video — playback is not blocked.
        </p>
      )}

      {phase === "error" && (
        <p role="alert" className="video-error">
          {PLAYBACK_ERROR}
        </p>
      )}

      {hasVideo ? (
        <div className="video-frame" ref={frameRef}>
          <video
            key={objectUrl}
            ref={videoRef}
            className="video-element"
            src={objectUrl}
            controls
            preload="metadata"
            playsInline
            aria-label="Local video player"
            onLoadedMetadata={handleLoadedMetadata}
            onCanPlay={() => {
              setBuffering(false);
              setPhase((current) => (current === "error" ? current : "ready"));
            }}
            onWaiting={() => setBuffering(true)}
            onPlaying={() => setBuffering(false)}
            onError={() => {
              setBuffering(false);
              setPhase("error");
            }}
          />
          {phase === "loading" && (
            <p role="status" className="video-overlay">
              Loading video metadata…
            </p>
          )}
          {buffering && phase === "ready" && (
            <p role="status" className="video-overlay">
              Buffering…
            </p>
          )}
        </div>
      ) : (
        <div className="video-empty">
          <p>No video selected.</p>
          <p className="video-empty-hint">
            Choose the original video above to play it and jump to evidence timestamps. Search works
            without a video.
          </p>
        </div>
      )}

      {hasVideo && phase !== "error" && (
        <div className="video-status">
          {formatHint && (
            <span className="video-format-hint">
              This browser reports it may not support {file?.type || "this format"}; playback will
              confirm.
            </span>
          )}
          {isValidDuration(videoDuration) && (
            <span className="video-duration">Duration {formatTimestamp(videoDuration)}</span>
          )}
          {seekedTo !== null && (
            <span role="status" className="video-seeked">
              Jumped to {formatTimestamp(seekedTo)}
            </span>
          )}
        </div>
      )}
    </section>
  );
});

export default VideoPlayer;