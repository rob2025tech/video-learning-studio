import { useEffect, useRef, useState, type RefObject } from "react";
import { ApiError, uploadReport } from "../api/client";
import { describeApiError } from "../api/messages";
import type { UploadResponse } from "../api/types";
import { formatBytes, formatDuration } from "../format";

type UploadState =
  | { kind: "idle" }
  | { kind: "selected"; file: File }
  | { kind: "uploading"; file: File }
  | { kind: "success"; session: UploadResponse }
  | { kind: "error"; file: File | null; message: string };

const GENERIC_FAILURE = "The upload failed for an unexpected reason. Try again.";

interface ReportUploadProps {
  onUploaded?: (session: UploadResponse) => void;
  /**
   * Shared monotonic upload counter owned by the app. Each startUpload claims the
   * next sequence; a response whose sequence is no longer current (a newer upload
   * started, possibly from a remounted panel) is discarded so it cannot overwrite
   * the newer session or clear its video.
   */
  uploadSeq?: RefObject<number>;
}

export default function ReportUpload({ onUploaded, uploadSeq }: ReportUploadProps) {
  const [state, setState] = useState<UploadState>({ kind: "idle" });
  const inputRef = useRef<HTMLInputElement>(null);
  const sessionHeadingRef = useRef<HTMLHeadingElement>(null);

  // Land keyboard/SR focus on the session summary once an upload succeeds.
  useEffect(() => {
    if (state.kind === "success") {
      sessionHeadingRef.current?.focus();
    }
  }, [state.kind]);

  const selectFile = (file: File | null | undefined) => {
    if (!file || state.kind === "uploading") {
      return;
    }
    setState({ kind: "selected", file });
  };

  const startUpload = async () => {
    const file =
      state.kind === "selected" || state.kind === "error" ? state.file : null;
    if (!file || state.kind === "uploading") {
      return;
    }
    const seq = uploadSeq ? ++uploadSeq.current : null;
    setState({ kind: "uploading", file });
    try {
      const session = await uploadReport(file);
      if (uploadSeq && seq !== uploadSeq.current) {
        return;
      }
      setState({ kind: "success", session });
      onUploaded?.(session);
    } catch (exc) {
      if (uploadSeq && seq !== uploadSeq.current) {
        return;
      }
      const message = exc instanceof ApiError ? describeApiError(exc) : GENERIC_FAILURE;
      setState({ kind: "error", file, message });
    }
  };

  const chooseAnother = () => {
    setState({ kind: "idle" });
    if (inputRef.current) {
      inputRef.current.value = "";
    }
  };

  if (state.kind === "success") {
    return (
      <div className="upload-panel">
        <SessionCard session={state.session} headingRef={sessionHeadingRef} />
        <button type="button" className="button" onClick={chooseAnother}>
          Choose another report
        </button>
      </div>
    );
  }

  const busy = state.kind === "uploading";
  const selectedFile = state.kind === "idle" ? null : state.file;
  const canUpload =
    state.kind === "selected" || (state.kind === "error" && state.file !== null);

  return (
    <section className="card upload-panel" aria-labelledby="upload-heading">
      <h2 id="upload-heading">Upload a report artifact</h2>
      <p id="upload-help" className="upload-help">
        Choose the JSON report created by <code>video-learning report</code> for a video
        you have already analyzed. LearnLens reads the report — the video itself stays on
        your machine and is never uploaded.
      </p>
      <div
        className="dropzone"
        onDragOver={(event) => event.preventDefault()}
        onDrop={(event) => {
          event.preventDefault();
          selectFile(event.dataTransfer.files[0]);
        }}
      >
        <label htmlFor="report-file">Report artifact (JSON)</label>
        <input
          ref={inputRef}
          id="report-file"
          type="file"
          aria-describedby="upload-help"
          disabled={busy}
          onChange={(event) => {
            selectFile(event.target.files?.[0]);
            // Allow re-selecting the same file after an error or a reset.
            event.target.value = "";
          }}
        />
        <p className="dropzone-hint">Drag and drop, or choose a file.</p>
      </div>
      {selectedFile && (
        <p className="file-name">
          Selected: <strong>{selectedFile.name}</strong>
        </p>
      )}
      {state.kind === "error" && (
        <p role="alert" className="error-text">
          {state.message}
        </p>
      )}
      {busy && (
        <p role="status" className="status-text">
          Uploading {state.file.name}…
        </p>
      )}
      <button type="button" className="button" disabled={!canUpload} onClick={startUpload}>
        {state.kind === "error" && state.file ? "Try again" : "Upload report"}
      </button>
    </section>
  );
}

function SessionCard({
  session,
  headingRef,
}: {
  session: UploadResponse;
  headingRef: RefObject<HTMLHeadingElement | null>;
}) {
  return (
    <section className="card session-summary" aria-label="Report session">
      <p role="status" className="success-text">
        Report loaded — session ready.
      </p>
      <h2 ref={headingRef} tabIndex={-1}>
        Report session
      </h2>
      <dl className="session-facts">
        <div>
          <dt>Report ID</dt>
          <dd>
            <code>{session.report_id}</code>
          </dd>
        </div>
        <div>
          <dt>File</dt>
          <dd>{session.filename}</dd>
        </div>
        <div>
          <dt>Size received</dt>
          <dd>{formatBytes(session.bytes_received)}</dd>
        </div>
        <div>
          <dt>Video duration</dt>
          <dd>
            {session.video_duration_seconds === null
              ? "Unknown"
              : formatDuration(session.video_duration_seconds)}
          </dd>
        </div>
        <div>
          <dt>Transcription</dt>
          <dd>
            {session.transcription_status}
            {session.transcription_available ? "" : " — speech evidence unavailable"}
          </dd>
        </div>
        <div>
          <dt>Session expiry</dt>
          <dd>
            ~{formatDuration(session.expires_in_seconds)} idle window from upload, renewed
            on use
          </dd>
        </div>
      </dl>
      <h3>Corpus</h3>
      <dl className="session-facts">
        <div>
          <dt>Records</dt>
          <dd>{session.corpus.record_count}</dd>
        </div>
        <div>
          <dt>Searchable records</dt>
          <dd>{session.corpus.searchable_record_count}</dd>
        </div>
        <div>
          <dt>Speech records</dt>
          <dd>{session.corpus.speech_record_count}</dd>
        </div>
        <div>
          <dt>OCR records</dt>
          <dd>{session.corpus.ocr_record_count}</dd>
        </div>
        <div>
          <dt>Segments</dt>
          <dd>{session.corpus.segment_count}</dd>
        </div>
      </dl>
    </section>
  );
}