"""Unit tests for InspectService: path validation and probe delegation."""

from __future__ import annotations

from pathlib import Path

import pytest

from video_learning.core.errors import FileNotFoundError_, NotAFileError
from video_learning.core.models import MediaInfo
from video_learning.services.inspect_service import InspectService


class FakeProbe:
    """Test double for the MediaProbe seam."""

    def __init__(self) -> None:
        self.probed: list[Path] = []

    def probe(self, path: Path) -> MediaInfo:
        self.probed.append(path)
        return MediaInfo(path=path, filename=path.name)


def test_inspect_returns_media_info(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"x")
    probe = FakeProbe()

    info = InspectService(probe=probe).inspect(video)

    assert info.filename == "clip.mp4"
    assert probe.probed == [video]


def test_inspect_missing_file_raises(tmp_path: Path) -> None:
    missing = tmp_path / "nope.mp4"
    probe = FakeProbe()

    with pytest.raises(FileNotFoundError_, match="File not found"):
        InspectService(probe=probe).inspect(missing)
    assert probe.probed == []


def test_inspect_directory_raises(tmp_path: Path) -> None:
    probe = FakeProbe()

    with pytest.raises(NotAFileError, match="Not a file"):
        InspectService(probe=probe).inspect(tmp_path)
    assert probe.probed == []
