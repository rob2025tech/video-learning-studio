"""One error shape for every API failure, so a client never has to parse a traceback.

Every refusal — ours or the framework's — is a JSON object with exactly two keys,
``code`` and ``detail``. FastAPI does not give that for free: its own 404/405/422 bodies
are ``{"detail": ...}``, and its default validation handler serializes ``exc.errors()``,
whose ``input`` field holds the offending request value verbatim. For a string carrying a
lone surrogate — which a JSON body can produce with an ASCII ``\\ud800`` escape — that
default render raises ``UnicodeEncodeError`` while building the error response, so a
malformed request becomes a 500 with a traceback.

:func:`http_error` and :func:`validation_error` therefore build their text from stable
identifiers only: a status number, a pydantic error type name, and field names drawn from
this API's own request models. No client-supplied string is copied into a detail, which
also means these functions cannot fail for the reasons the framework's handlers do.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

#: One machine-readable code per framework status this API can return, so a client can
#: branch on ``code`` whichever layer produced the refusal. An unmapped status keeps a
#: deterministic code derived from the number rather than from anything in the request.
_STATUS_CODES: dict[int, str] = {
    400: "bad_request",
    404: "route_not_found",
    405: "method_not_allowed",
    413: "request_too_large",
    415: "unsupported_media_type",
    422: "invalid_request",
}

#: Fixed wording per status. Deliberately terse: Starlette's own details ("Not Found", a
#: parser's byte-level diagnostic, the requested path) are either uninformative or
#: request-derived, and a 404 in particular is reachable for any path a client invents.
_STATUS_DETAILS: dict[int, str] = {
    400: "The request could not be processed.",
    404: "No API route is served at this path.",
    405: "That method is not allowed on this route.",
    413: "The request body exceeds the limit this API is configured with.",
    415: "This route accepts only the content types it documents.",
    422: "The request did not validate.",
}

#: ``body`` is the location pydantic uses for the request body; the rest are the fields
#: of :class:`~video_learning.api.schemas.SearchRequest`. Only a name in this closed set
#: can reach a detail, so an unexpected key — or one carrying a lone surrogate — is
#: reported as ``body`` instead of quoted back.
KNOWN_REQUEST_FIELDS: frozenset[str] = frozenset({"body", "report_id", "query", "limit"})

#: How many field errors the detail names before switching to a bare count. A validation
#: failure normally produces one or two; a flood would only crowd out the useful part.
MAX_DETAIL_FIELDS = 4


class ApiError(Exception):
    """A refusal with a stable machine-readable ``code`` and a human-readable ``detail``."""

    def __init__(self, status_code: int, code: str, detail: str) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.code = code
        self.detail = detail

    def to_dict(self) -> dict[str, str]:
        return {"code": self.code, "detail": self.detail}


def http_error(status_code: int) -> ApiError:
    """Map a framework HTTP error onto this API's shape, echoing nothing from the request."""
    return ApiError(
        status_code,
        _STATUS_CODES.get(status_code, f"http_{status_code}"),
        _STATUS_DETAILS.get(status_code, "The request was refused."),
    )


def validation_error(errors: Sequence[Any]) -> ApiError:
    """Summarize a pydantic validation failure without touching its ``input`` values.

    Each entry names one field and one error type. The field comes from the location
    tuple's last element, and only if it is a name this API actually declares — so an
    unexpected key, an array index and anything else the client invented all read as
    ``body``. The type is pydantic's own stable identifier, which is what a client can
    branch on; the human-readable ``msg`` is dropped because pydantic interpolates values
    into some messages.
    """
    entries = list(errors)
    named: list[str] = []
    for entry in entries[:MAX_DETAIL_FIELDS]:
        item: Mapping[str, Any] = entry if isinstance(entry, Mapping) else {}
        field = next(
            (
                part
                for part in reversed(tuple(item.get("loc", ())))
                if isinstance(part, str) and part in KNOWN_REQUEST_FIELDS
            ),
            "body",
        )
        kind = item.get("type")
        if not isinstance(kind, str) or not kind.isidentifier():
            kind = "invalid"
        named.append(f"{field}: {kind}")
    if len(entries) > MAX_DETAIL_FIELDS:
        named.append("…")
    detail = (
        f"Request body did not validate ({len(entries)} field error(s)): {', '.join(named)}."
        if named
        else "Request body did not validate."
    )
    return ApiError(422, "invalid_request", detail)
