"""No HTTP route may run media processing, or touch the filesystem at all.

``subprocess`` is patched to raise, which is the same technique the search CLI tests use:
it turns "the API ran no pipeline" into a failure instead of a claim. The filesystem
patch is stronger still — an API that reads only the request stream should survive having
every ``Path`` read refuse, which also proves the uploaded filename is never resolved.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from tests.unit.api_support import client_for, report_id, search, upload

_NO_SUBPROCESS = patch.object(
    subprocess, "run", side_effect=AssertionError("the API must not run a subprocess")
)
_NO_POPEN = patch.object(
    subprocess, "Popen", side_effect=AssertionError("the API must not spawn a process")
)


def _no_path_reads() -> Any:
    """Refuse every pathlib read, so a route cannot open the artifact, the video, or a name."""
    return (
        patch.object(Path, "read_text", side_effect=AssertionError("no read_text")),
        patch.object(Path, "read_bytes", side_effect=AssertionError("no read_bytes")),
        patch.object(Path, "open", side_effect=AssertionError("no Path.open")),
    )


@pytest.fixture
def client() -> TestClient:
    return client_for()


def test_every_route_runs_without_a_subprocess(
    client: TestClient, report_artifact_bytes: bytes
) -> None:
    with _NO_SUBPROCESS, _NO_POPEN:
        assert client.get("/api/health").status_code == 200
        assert client.get("/api/openapi.json").status_code == 200
        assert client.get("/api/docs").status_code == 200
        rid = report_id(client, report_artifact_bytes)
        assert search(client, rid).status_code == 200
        assert search(client, rid, "", limit=0).status_code == 200
        assert search(client, rid, "zzzznotpresent").status_code == 200
        assert search(client, "unknown").status_code == 404


def test_a_refused_upload_runs_no_subprocess(client: TestClient) -> None:
    with _NO_SUBPROCESS, _NO_POPEN:
        assert upload(client, b"{not json").status_code == 400
        assert upload(client, b'{"schema": "other/v1"}').status_code == 400
        assert client.post("/api/reports").status_code == 422


def test_the_whole_flow_needs_no_filesystem_read(
    client: TestClient, report_artifact_bytes: bytes
) -> None:
    """Upload, search and health all work with every ``Path`` read patched to raise.

    The artifact arrives as request bytes and the video is never opened, so there is no
    path left for a route to read. The fixture's own write happened before the patch.
    """
    read_text, read_bytes, open_path = _no_path_reads()
    with _NO_SUBPROCESS, _NO_POPEN, read_text, read_bytes, open_path:
        rid = report_id(client, report_artifact_bytes)
        payload = search(client, rid, "index", limit=100).json()
        assert payload["coverage"]["matched_record_count"] == 7
        assert client.get("/api/health").json()["active_report_sessions"] == 1


def test_the_video_path_in_the_artifact_is_never_opened(
    client: TestClient, report_artifact_bytes: bytes, tmp_path: Path
) -> None:
    """A sentinel file stands in for the video: opening it would trip the guard."""
    video = tmp_path / "clip.mov"
    video.write_bytes(b"not a video, and never read")
    opened: list[Path] = []
    real_open = Path.open

    def _watch(self: Path, *args: Any, **kwargs: Any) -> Any:
        opened.append(self)
        return real_open(self, *args, **kwargs)

    with _NO_SUBPROCESS, _NO_POPEN, patch.object(Path, "open", _watch):
        rid = report_id(client, report_artifact_bytes)
        assert search(client, rid).status_code == 200

    assert video not in opened