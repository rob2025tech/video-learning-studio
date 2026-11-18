"""Upload handling: untrusted bytes in, one validated ``SearchService`` out.

Validation is entirely the existing store's. ``ReportStore.load_evidence_from_text``
already checks JSON syntax, the object shape, the ``video-learning.report/v1`` schema
string, the segmentation assembly constants and every timestamp type, and it already
turns each failure into one ``ReportArtifactError``. This module adds no second
validator and no reinterpretation of those rules — it only supplies the text and maps
the error onto an HTTP status. The one extra gate is about transport rather than content:
:func:`_require_utf8_transport` refuses a document whose strings this API could never
send back as UTF-8, and it runs only after the store has already accepted the artifact.

Nothing here opens a path. The uploaded bytes are read from the request stream, and the
multipart filename is reduced to display metadata by
:func:`~video_learning.api.responses.safe_display_name` before it is stored or shown.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from fastapi import Request
from starlette.datastructures import UploadFile
from starlette.exceptions import HTTPException as StarletteHTTPException

from video_learning.api.errors import ApiError
from video_learning.api.responses import safe_display_name
from video_learning.api.sessions import ReportSession, ReportSessionStore
from video_learning.core.errors import ReportArtifactError
from video_learning.services.report_service import REPORT_SCHEMA, ReportStore
from video_learning.services.search_service import SearchService

REPORT_FIELD = "report"
FALLBACK_REPORT_NAME = "report.json"

#: Stable detail for a rejected artifact. The store's own message quotes the uploaded
#: filename and artifact-derived values (the schema string found, the parse position),
#: so it is logged server-side and kept out of the response entirely.
INVALID_REPORT_DETAIL = "The report artifact is invalid."

_logger = logging.getLogger(__name__)


async def read_report_upload(request: Request, *, max_bytes: int) -> tuple[str, str, int]:
    """Return ``(text, display_filename, byte_count)`` for the one uploaded report.

    ``max_part_size`` is raised to the request cap so a legitimate multi-megabyte report
    is not rejected by the parser's 1 MiB default; the hard bound is enforced upstream by
    :class:`~video_learning.api.security.BodyLimitMiddleware`, which counts actual bytes.
    """
    try:
        form = await request.form(max_files=1, max_fields=0, max_part_size=max_bytes)
    except StarletteHTTPException as exc:
        raise ApiError(400, "invalid_multipart", str(exc.detail)) from exc
    try:
        upload = form.get(REPORT_FIELD)
        # Starlette's class, not fastapi.UploadFile: the latter is a subclass, and the
        # multipart parser constructs the base, so isinstance against FastAPI's fails.
        if not isinstance(upload, UploadFile):
            raise ApiError(
                422,
                "missing_report",
                f"Expected one multipart file field named {REPORT_FIELD!r}.",
            )
        data = await upload.read()
        filename = safe_display_name(upload.filename, fallback=FALLBACK_REPORT_NAME)
    finally:
        await form.close()
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ApiError(400, "invalid_encoding", "The report artifact must be UTF-8 text.") from exc
    return text, filename, len(data)


def open_session(
    store: ReportSessionStore, *, text: str, filename: str, byte_count: int
) -> ReportSession:
    """Validate the artifact, build its index once, and retain it as a session.

    The UTF-8 transport check runs between validation and retention: an artifact holding a
    lone surrogate is refused before ``store.create``, so no slot is taken and no half-built
    session exists to be queried. See :func:`_require_utf8_transport` for why that belongs
    here rather than in the route.
    """
    try:
        evidence = ReportStore().load_evidence_from_text(text, label=filename)
    except ReportArtifactError as exc:
        _logger.warning("rejected uploaded report artifact (%s): %s", exc.__class__.__name__, exc)
        raise ApiError(400, "invalid_report", INVALID_REPORT_DETAIL) from exc
    _require_utf8_transport(text)
    return store.create(
        service=SearchService(evidence),
        filename=filename,
        segment_count=len(evidence.segments),
        bytes_in=byte_count,
    )


def _require_utf8_transport(text: str) -> None:
    """Refuse an artifact whose parsed strings cannot be encoded as UTF-8.

    A JSON body can carry a lone surrogate in pure ASCII — ``\\ud800`` is two backslash
    characters here and an unpaired half of a surrogate pair once parsed — so the request
    bytes, and the decoded text, both look clean until the document is parsed. UTF-8 has
    no encoding for a surrogate, and every response this API sends is UTF-8 (Starlette
    renders JSON with ``ensure_ascii=False``), so accepting such a document would mean
    either a 500 partway through a search, after the session is already retained, or
    evidence text silently rewritten. This refuses instead, once, before retention: the
    400 says what is wrong, and the CLI keeps reading the same bytes because it writes
    with ``ensure_ascii=True`` and never needed a UTF-8 response.

    It runs only after the store has accepted the artifact, so a malformed report is still
    refused by the store's rules alone and this adds no second set of artifact rules; the
    two refusals share the one stable ``invalid_report`` response detail either way.
    """
    try:
        json.dumps(json.loads(text), ensure_ascii=False).encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ApiError(
            400, "invalid_report", "The report artifact holds text that cannot be sent as UTF-8."
        ) from exc


def upload_payload(session: ReportSession, *, store: ReportSessionStore) -> dict[str, Any]:
    """Describe the loaded report using only what the retained index already knows.

    No query is run to produce this: the corpus counts and the transcription state are
    read straight off ``SearchService.index``, so the UI can say "7 records, transcription
    unavailable" before the user types anything.
    """
    index = session.service.index
    return {
        "report_id": session.report_id,
        "report_schema": REPORT_SCHEMA,
        "filename": session.filename,
        "bytes_received": session.bytes_in,
        "expires_in_seconds": store.expires_in(session),
        "video_duration_seconds": index.video_duration_seconds,
        "transcription_status": index.transcription_status,
        "transcription_available": index.transcription_available,
        "corpus": {
            "record_count": len(index.records),
            "searchable_record_count": len(index.searchable_positions),
            "speech_record_count": sum(
                1 for record in index.records if record.source_type == "speech"
            ),
            "ocr_record_count": sum(
                1 for record in index.records if record.source_type == "ocr"
            ),
            "segment_count": session.segment_count,
        },
    }
