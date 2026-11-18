"""Integration test: the CLI in an environment where the API stack is genuinely absent.

This is not an import blocker and not a ``sys.modules`` check. A second virtual
environment is synchronized from the locked project with the dev group omitted
(``uv sync --frozen --no-dev --offline``), redirected away from the project's ``.venv``
by ``UV_PROJECT_ENVIRONMENT``. That environment contains typer, rich and their
transitive dependencies plus this project — and no fastapi, starlette, pydantic,
uvicorn, multipart or httpx at all. The test asserts their absence with
``importlib.util.find_spec`` returning ``None``, then runs the real ``video-learning``
console script from that environment.

Environmental assumptions, stated rather than hidden:

* ``uv`` must be on PATH, and the local uv cache must already hold the base dependencies
  and the build backend. ``--offline`` is deliberate: this test must never reach the
  network. When either assumption fails the module skips with the reason, following the
  same convention as the real-media integration tests.
* ``--frozen`` means the lockfile is read, never rewritten.
* The project's ``.venv`` is fingerprinted before and after the sync and compared, so
  "did not touch the active environment" is an assertion rather than an intention.

Everything created lives in one ``TemporaryDirectory`` that is removed on teardown,
including the artifact the search command reads.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pytest

from tests.conftest import synthetic_report
from video_learning import __version__
from video_learning.services.report_service import ReportStore
from video_learning.services.search_service import SearchService

REPO_ROOT = Path(__file__).resolve().parents[2]
PROJECT_VENV = REPO_ROOT / ".venv"

#: Every importable name the API extra pulls in, including the compiled/renamed halves.
ABSENT_MODULES = (
    "fastapi",
    "starlette",
    "pydantic",
    "pydantic_core",
    "uvicorn",
    "multipart",
    "python_multipart",
    "httpx",
    "anyio",
)
BASE_MODULES = ("typer", "rich", "video_learning", "video_learning.cli.main")

pytestmark = pytest.mark.skipif(shutil.which("uv") is None, reason="requires uv on PATH")


@dataclass(frozen=True)
class IsolatedCli:
    """The throwaway environment, the artifact it searches, and the pre-sync fingerprint."""

    python: Path
    script: Path
    web_script: Path
    artifact: Path
    venv_before: tuple[bytes, tuple[str, ...]]


def _venv_fingerprint() -> tuple[bytes, tuple[str, ...]]:
    """Identity of the active environment: its config plus its installed distributions."""
    config = (PROJECT_VENV / "pyvenv.cfg").read_bytes()
    distributions = tuple(
        sorted(path.name for path in PROJECT_VENV.glob("lib/*/site-packages/*.dist-info"))
    )
    return config, distributions


def _cli_env() -> dict[str, str]:
    """A minimal environment: no VIRTUAL_ENV or PYTHONPATH leaking the project venv in."""
    return {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": os.environ.get("HOME", ""),
        "PYTHONDONTWRITEBYTECODE": "1",
        "COLUMNS": "200",
        "NO_COLOR": "1",
    }


def _run(command: list[str], *, timeout: int = 120) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command, capture_output=True, text=True, timeout=timeout, env=_cli_env(), check=False
    )


@pytest.fixture(scope="module")
def isolated() -> Iterator[IsolatedCli]:
    before = _venv_fingerprint()
    with tempfile.TemporaryDirectory(prefix="vls-cli-only-") as scratch:
        root = Path(scratch)
        venv = root / "venv"
        env = dict(os.environ, UV_PROJECT_ENVIRONMENT=str(venv))
        env.pop("VIRTUAL_ENV", None)
        synced = subprocess.run(
            ["uv", "sync", "--frozen", "--no-dev", "--offline"],
            cwd=REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=600,
            check=False,
        )
        if synced.returncode != 0:
            pytest.skip(
                "could not build an isolated CLI-only environment offline "
                f"(uv exit {synced.returncode}): {synced.stderr.strip()[-400:]}"
            )

        artifact = root / "report.json"
        ReportStore().save(synthetic_report(root / "clip.mov"), artifact)
        yield IsolatedCli(
            python=venv / "bin" / "python",
            script=venv / "bin" / "video-learning",
            web_script=venv / "bin" / "video-learning-web",
            artifact=artifact,
            venv_before=before,
        )


def test_the_api_stack_is_genuinely_absent_and_the_base_is_present(isolated: IsolatedCli) -> None:
    names = list(ABSENT_MODULES + BASE_MODULES)
    probe = (
        "import importlib.util as u, json;"
        f"print(json.dumps({{m: u.find_spec(m) is None for m in {names!r}}}))"
    )
    result = _run([str(isolated.python), "-c", probe])

    assert result.returncode == 0, result.stderr
    absent = json.loads(result.stdout)
    for module in ABSENT_MODULES:
        assert absent[module] is True, f"{module} is installed in the isolated environment"
    for module in BASE_MODULES:
        assert absent[module] is False, f"{module} is missing from the isolated environment"


def test_the_version_command_runs(isolated: IsolatedCli) -> None:
    result = _run([str(isolated.script), "--version"])

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == f"video-learning {__version__}"


def test_search_json_equals_the_cores_answer(isolated: IsolatedCli) -> None:
    """The whole payload, including the unredacted absolute source the CLI has always shown."""
    result = _run([str(isolated.script), "search", str(isolated.artifact), "index", "--json"])

    assert result.returncode == 0, result.stderr
    expected = (
        SearchService(ReportStore().load_evidence(isolated.artifact))
        .query("index", limit=5)
        .to_dict()
    )
    assert json.loads(result.stdout) == expected


def test_search_limit_zero_still_reports_counts(isolated: IsolatedCli) -> None:
    result = _run(
        [str(isolated.script), "search", str(isolated.artifact), "index", "--limit", "0", "--json"]
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["status"] == "ok"
    assert payload["result_count"] == 0
    assert payload["coverage"]["matched_record_count"] == 7


def test_the_readable_search_render_runs(isolated: IsolatedCli) -> None:
    result = _run([str(isolated.script), "search", str(isolated.artifact), "index"])

    assert result.returncode == 0, result.stderr
    assert "SEARCH" in result.stdout
    assert "the video was not opened" in result.stdout


def test_importing_the_cli_imports_nothing_from_the_api_stack(isolated: IsolatedCli) -> None:
    names = list(ABSENT_MODULES)
    probe = (
        "import sys, json;"
        "import video_learning.cli.main;"
        f"print(json.dumps(sorted(m for m in sys.modules if m.split('.')[0] in {names!r})))"
    )
    result = _run([str(isolated.python), "-c", probe])

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == []


def test_the_web_entry_point_is_the_one_that_needs_the_extra(isolated: IsolatedCli) -> None:
    """The boundary in the other direction: without the extra, the API entry point fails."""
    result = _run([str(isolated.web_script), "--help"])

    assert result.returncode != 0
    assert "ModuleNotFoundError" in result.stderr
    assert "fastapi" in result.stderr


def test_the_project_venv_was_not_modified(isolated: IsolatedCli) -> None:
    assert _venv_fingerprint() == isolated.venv_before


def test_the_sync_left_no_build_artifacts_in_the_repository(isolated: IsolatedCli) -> None:
    """The editable install belongs to the throwaway venv, not to the working tree.

    Cleanup of the environment itself is the fixture's ``TemporaryDirectory`` context,
    which removes the venv and the artifact even when a test fails.
    """
    assert isolated.python.exists()
    assert sorted(path.name for path in REPO_ROOT.glob("*.egg-info")) == []
    assert sorted(path.name for path in REPO_ROOT.glob("src/*.egg-info")) == []
    assert not (REPO_ROOT / "build").exists()
    assert not (REPO_ROOT / ".venv-cli-only").exists()