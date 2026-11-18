"""Localhost-only request policy, and a body bound enforced on the stream itself.

Two independent controls, because neither alone is sufficient:

* **The Host allowlist is what defeats DNS rebinding.** A page at ``evil.example`` whose
  DNS answer is ``127.0.0.1`` still sends ``Host: evil.example``. CORS does not stop that
  request from reaching a handler — it only stops the attacker *reading* the response —
  and a state-changing POST would already have run. Rejecting the foreign Host does stop
  it. This is why the policy checks Host on every request, including GETs.
* **The body limit counts bytes off the ASGI receive stream**, not ``Content-Length``, so
  a client that omits or understates the header cannot make the server buffer more than
  the cap. ``Content-Length`` is advisory; the stream is the truth. The count happens
  before the app is called, so the refusal does not depend on how a route parses a body.

Missing ``Origin`` is allowed on purpose: curl and same-origin top-level navigations omit
it, while a browser always sends it on a cross-origin fetch. Absence therefore cannot
indicate a cross-origin read, and the Host check already covers rebinding.
"""

from __future__ import annotations

from dataclasses import dataclass

from starlette.datastructures import Headers
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

#: Fixed refusal texts. Deliberately non-reflective: interpolating the client's Host or
#: Origin quotes untrusted request data back in an error body, which adds no diagnostic
#: value a log entry could not provide and turns the 403 into an echo surface. The codes
#: already tell a client which check failed.
_HOST_REJECTION_DETAIL = "Request Host is not allowed."
_HOST_PORT_REJECTION_DETAIL = "Request Host port is not allowed."
_HOST_MISSING_PORT_REJECTION_DETAIL = "Request Host omits the port this API listens on."
_ORIGIN_REJECTION_DETAIL = "Request Origin is not allowed."

#: The server binds 127.0.0.1 only, so these are the two names a legitimate client can
#: have used to reach it. ``localhost`` is included because browsers prefer it and it
#: resolves to the loopback address we actually listen on.
ALLOWED_HOST_NAMES: tuple[str, ...] = ("127.0.0.1", "localhost")
DEFAULT_PORT = 8000
#: The only port for which a port-less Host header is legitimate.
_IMPLICIT_HTTP_PORT = 80


@dataclass(frozen=True)
class RequestPolicy:
    """The exact localhost contract, as a pure function of the request headers."""

    port: int = DEFAULT_PORT

    @property
    def allowed_origins(self) -> tuple[str, ...]:
        return tuple(f"http://{name}:{self.port}" for name in ALLOWED_HOST_NAMES)

    def rejection(self, headers: Headers) -> tuple[str, str] | None:
        """Return ``(code, detail)`` when the request must be refused, else ``None``."""
        host = headers.get("host", "")
        name, separator, port_text = host.partition(":")
        if name.lower() not in ALLOWED_HOST_NAMES:
            return ("forbidden_host", _HOST_REJECTION_DETAIL)
        if separator:
            # isascii() first: str.isdigit() is true for latin-1 superscripts such as "²",
            # which int() then refuses with a ValueError. Header values are latin-1, so
            # that byte can genuinely arrive; without this guard it would be a 500.
            if not port_text.isascii() or not port_text.isdigit() or int(port_text) != self.port:
                return ("forbidden_host", _HOST_PORT_REJECTION_DETAIL)
        elif self.port != _IMPLICIT_HTTP_PORT:
            return ("forbidden_host", _HOST_MISSING_PORT_REJECTION_DETAIL)

        origin = headers.get("origin")
        if origin is not None and origin not in self.allowed_origins:
            return ("forbidden_origin", _ORIGIN_REJECTION_DETAIL)
        if headers.get("sec-fetch-site") == "cross-site":
            return ("forbidden_origin", "Cross-site requests are not accepted.")
        return None


class HostOriginMiddleware:
    """Refuse foreign Host/Origin before any handler runs or any body is read."""

    def __init__(self, app: ASGIApp, *, port: int = DEFAULT_PORT) -> None:
        self.app = app
        self.policy = RequestPolicy(port=port)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        rejection = self.policy.rejection(Headers(scope=scope))
        if rejection is not None:
            code, detail = rejection
            response = JSONResponse({"code": code, "detail": detail}, status_code=403)
            await response(scope, receive, send)
            return
        await self.app(scope, receive, send)


class BodyLimitMiddleware:
    """Hard-cap the request body at ``max_bytes``, regardless of what headers claim.

    The body is buffered here, before the app is called, and the 413 is sent from here
    when the cap is crossed. Counting inside a ``receive`` wrapper instead does not work:
    Starlette short-circuits ``Request.form()`` to an empty form without reading the
    stream when the content type is not multipart, and FastAPI wraps body parsing in a
    broad ``except Exception`` that would turn our refusal into a 400. Owning the read
    makes the bound independent of both.

    Buffering costs at most ``max_bytes`` per in-flight request, which is the same order
    the multipart parser would have held anyway. An oversized request is refused without
    the app ever running, so no session is created and no handler sees a partial body.
    """

    def __init__(self, app: ASGIApp, *, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        buffered: list[Message] = []
        total = 0
        while True:
            message = await receive()
            buffered.append(message)
            if message["type"] != "http.request":
                # http.disconnect, or anything else the server injects: stop reading.
                break
            total += len(message.get("body", b""))
            if total > self.max_bytes:
                response = JSONResponse(
                    {
                        "code": "request_too_large",
                        "detail": f"Request body exceeds the {self.max_bytes}-byte limit.",
                    },
                    status_code=413,
                )
                await response(scope, receive, send)
                return
            if not message.get("more_body", False):
                break

        position = 0

        async def replay() -> Message:
            nonlocal position
            if position < len(buffered):
                message = buffered[position]
                position += 1
                return message
            return await receive()

        await self.app(scope, replay, send)
