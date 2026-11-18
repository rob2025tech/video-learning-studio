"""The local evidence API: three routes over the existing search services.

Composition root for the HTTP surface, mirroring what ``cli/main.py`` does for the
terminal: construct collaborators, call a service, serialize. The search route is the
same two calls the ``search`` command makes — ``ReportStore`` evidence in,
``SearchService.query`` out — with the index built once at upload instead of per query.

Deliberately absent: any route that runs ``ReportService.build`` (that would put ffmpeg,
tesseract and whisper.cpp behind an HTTP request), any video or media route (the browser
plays a local file through a Blob URL and never uploads it), any database, and any
persistent report storage. Sessions live in memory and die with the process.

The server binds 127.0.0.1 only and there is no ``--host`` override: this API reads a
user's local evidence and has no authentication, so it must not be reachable from the
network. See :mod:`video_learning.api.security` for the Host/Origin policy that keeps a
foreign page from driving it even on loopback.
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException as StarletteHTTPException

from video_learning import __version__
from video_learning.api.errors import ApiError, http_error, validation_error
from video_learning.api.reports import open_session, read_report_upload, upload_payload
from video_learning.api.responses import search_payload
from video_learning.api.schemas import SearchRequest
from video_learning.api.security import (
    DEFAULT_PORT,
    BodyLimitMiddleware,
    HostOriginMiddleware,
)
from video_learning.api.sessions import (
    DEFAULT_HARD_TTL_SECONDS,
    DEFAULT_IDLE_TTL_SECONDS,
    DEFAULT_MAX_SESSIONS,
    ReportSessionStore,
)
from video_learning.core.search import SEARCH_SCHEMA
from video_learning.services.report_service import REPORT_SCHEMA

LOOPBACK_HOST = "127.0.0.1"
#: An input bound on the uploaded artifact, not a memory ceiling. See api/sessions.py.
DEFAULT_MAX_REPORT_BYTES = 8 * 1024 * 1024


def create_app(
    *,
    port: int = DEFAULT_PORT,
    max_sessions: int = DEFAULT_MAX_SESSIONS,
    max_report_bytes: int = DEFAULT_MAX_REPORT_BYTES,
    idle_ttl_seconds: float = DEFAULT_IDLE_TTL_SECONDS,
    hard_ttl_seconds: float = DEFAULT_HARD_TTL_SECONDS,
    clock: Callable[[], float] = time.monotonic,
) -> FastAPI:
    """Build the app. ``clock`` is injectable so session expiry is testable without sleeps."""
    app = FastAPI(
        title="LearnLens local evidence API",
        version=__version__,
        docs_url="/api/docs",
        openapi_url="/api/openapi.json",
        redoc_url=None,
        # No OAuth2 flow exists here, so FastAPI's default redirect route would be a
        # reachable endpoint with nothing behind it.
        swagger_ui_oauth2_redirect_url=None,
    )
    store = ReportSessionStore(
        max_sessions=max_sessions,
        idle_ttl_seconds=idle_ttl_seconds,
        hard_ttl_seconds=hard_ttl_seconds,
        clock=clock,
    )
    app.state.store = store

    @app.exception_handler(ApiError)
    async def _api_error(request: Request, exc: ApiError) -> JSONResponse:
        return JSONResponse(exc.to_dict(), status_code=exc.status_code)

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        # Starlette raises this for an unknown path (404) and a known path reached by the
        # wrong method (405, with an Allow header). Without the handler the body would be
        # ``{"detail": ...}``, so a client would have to parse two shapes.
        error = http_error(exc.status_code)
        return JSONResponse(error.to_dict(), status_code=error.status_code, headers=exc.headers)

    @app.exception_handler(RequestValidationError)
    async def _validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        # FastAPI's default handler serializes exc.errors(), whose "input" field is the
        # rejected value itself. A surrogate-bearing query therefore raises while rendering
        # this very error; validation_error names fields and types, never values.
        error = validation_error(exc.errors())
        return JSONResponse(error.to_dict(), status_code=error.status_code)

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        store.purge_expired()
        return {
            "status": "ok",
            "version": __version__,
            "report_schema": REPORT_SCHEMA,
            "search_schema": SEARCH_SCHEMA,
            "active_report_sessions": len(store),
            "max_report_sessions": store.max_sessions,
            "max_report_bytes": max_report_bytes,
        }

    @app.post("/api/reports", status_code=201)
    async def upload_report(request: Request) -> dict[str, Any]:
        text, filename, byte_count = await read_report_upload(
            request, max_bytes=max_report_bytes
        )
        # Parsing and indexing are synchronous and cost scales with the artifact, so they
        # run in a worker thread: an ``async def`` route body owns the event loop, and
        # calling them inline would stall every other in-flight request. The multipart
        # read above stays on the loop, where awaiting a stream is the right shape.
        session = await run_in_threadpool(
            open_session, store, text=text, filename=filename, byte_count=byte_count
        )
        return upload_payload(session, store=store)

    @app.post("/api/search")
    def search_reports(payload: SearchRequest) -> dict[str, Any]:
        # Sync on purpose: FastAPI runs it in a worker thread, so a lexical pass over a
        # large corpus never blocks the event loop. The query itself is pure and reads a
        # frozen index, so no lock is needed around it.
        session = store.get(payload.report_id)
        results = session.service.query(payload.query, limit=payload.limit)
        return search_payload(results)

    # add_middleware wraps outward, so the last one added runs first: Host/Origin is
    # checked before a single body byte is read.
    app.add_middleware(BodyLimitMiddleware, max_bytes=max_report_bytes)
    app.add_middleware(HostOriginMiddleware, port=port)
    return app


def main() -> None:
    """Console entry point: ``video-learning-web [--port N]``, loopback only."""
    parser = argparse.ArgumentParser(
        prog="video-learning-web",
        description="Serve the LearnLens evidence API on 127.0.0.1 (requires the api extra).",
    )
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Loopback port to bind.")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error(f"--port must be between 1 and 65535, got {args.port}")

    import uvicorn

    uvicorn.run(create_app(port=args.port), host=LOOPBACK_HOST, port=args.port)
