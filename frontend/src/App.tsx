import { useRef, useState } from "react";
import type { UploadResponse } from "./api/types";
import EvidenceSearch from "./components/EvidenceSearch";
import ReportUpload from "./components/ReportUpload";
import VideoPlayer, { type VideoPlayerHandle } from "./components/VideoPlayer";

type View = "workspace" | "upload" | "search";

export default function App() {
  const [view, setView] = useState<View>("workspace");
  const [session, setSession] = useState<UploadResponse | null>(null);
  // The selected local video is owned here, keyed to the active report, so it survives
  // switching between Workspace and Evidence search. The object URL itself lives in
  // VideoPlayer and is recreated from this File on remount.
  const [videoFile, setVideoFile] = useState<File | null>(null);
  const [selectedEvidenceId, setSelectedEvidenceId] = useState<string | null>(null);
  const videoPlayerRef = useRef<VideoPlayerHandle>(null);
  // Shared across ReportUpload mounts so an older in-flight upload cannot
  // overwrite a newer session (see ReportUpload uploadSeq).
  const uploadSeqRef = useRef(0);

  const handleUploaded = (uploaded: UploadResponse) => {
    setSession(uploaded);
    // A new report invalidates any video selected against the previous one.
    setVideoFile(null);
    setSelectedEvidenceId(null);
  };

  const handleSeek = (seconds: number, evidenceId: string) => {
    // Only mark a result selected when the player actually honoured the seek;
    // a missing or unplayable video must not leave a stale "Selected" card.
    const sought = videoPlayerRef.current?.seek(seconds) ?? false;
    setSelectedEvidenceId(sought ? evidenceId : null);
  };

  const handleSelectVideo = (file: File) => {
    setVideoFile(file);
    // A replacement video starts at 0:00; any prior selection no longer matches.
    setSelectedEvidenceId(null);
  };

  const handleClearVideo = () => {
    setVideoFile(null);
    setSelectedEvidenceId(null);
  };

  return (
    <div className="app">
      <header className="app-header">
        <div className="brand">
          <h1>LearnLens</h1>
          <p className="tagline">Local-first video study evidence</p>
        </div>
        <nav aria-label="Primary">
          <ul className="nav-list">
            <li>
              <button
                type="button"
                className={`nav-link${view === "workspace" ? " is-active" : ""}`}
                aria-current={view === "workspace" ? "page" : undefined}
                onClick={() => setView("workspace")}
              >
                Workspace
              </button>
            </li>
            <li>
              <button
                type="button"
                className={`nav-link${view === "upload" ? " is-active" : ""}`}
                aria-current={view === "upload" ? "page" : undefined}
                onClick={() => setView("upload")}
              >
                Report upload
              </button>
            </li>
            <li>
              {session ? (
                <button
                  type="button"
                  className={`nav-link${view === "search" ? " is-active" : ""}`}
                  aria-current={view === "search" ? "page" : undefined}
                  onClick={() => setView("search")}
                >
                  Evidence search
                </button>
              ) : (
                <span className="nav-link is-disabled" aria-disabled="true" title="Upload a report first">
                  Evidence search
                </span>
              )}
            </li>
          </ul>
        </nav>
      </header>
      <main className="workspace">
        {view === "workspace" && (
          <section className="card workspace-intro">
            <h2>Video learning, grounded in evidence</h2>
            <p>
              LearnLens turns a local tutorial video into a searchable study record: spoken
              words and on-screen text, each tied to the timestamp where it appears. Search
              returns the exact evidence and the moment to jump to — nothing is summarized
              away.
            </p>
            <p>
              {session
                ? `“${session.filename}” is loaded. Open Evidence search to query its ${session.corpus.record_count} records.`
                : "Start by uploading a report artifact from the Report upload section, then search its evidence."}
            </p>
          </section>
        )}
        {view === "upload" && (
          <ReportUpload onUploaded={handleUploaded} uploadSeq={uploadSeqRef} />
        )}
        {/* Kept mounted (hidden) across view switches so query, results, playback
            position and seek feedback survive navigation; no state library needed. */}
        {session && (
          <div className="search-view" hidden={view !== "search"}>
            <VideoPlayer
              ref={videoPlayerRef}
              file={videoFile}
              reportDurationSeconds={session.video_duration_seconds}
              onSelect={handleSelectVideo}
              onClear={handleClearVideo}
              active={view === "search"}
            />
            <EvidenceSearch
              session={session}
              onNavigateUpload={() => setView("upload")}
              onSeek={handleSeek}
              selectedEvidenceId={selectedEvidenceId}
              hasVideo={videoFile !== null}
            />
          </div>
        )}
      </main>
      <footer className="app-footer">
        Everything runs on your machine. No accounts, no cloud.
      </footer>
    </div>
  );
}