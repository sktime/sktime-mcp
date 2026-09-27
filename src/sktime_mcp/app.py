"""HTTP/SSE front-end for the MCP server (``uvicorn sktime_mcp.app:app``).

Every request except the ``/`` health check must carry
``Authorization: Bearer <SKTIME_MCP_HTTP_TOKEN>``; Host and Origin headers are
checked against an allow-list (DNS-rebinding protection and CORS); and
``run_command`` is hidden from HTTP clients unless explicitly re-enabled. The
stdio server in :mod:`sktime_mcp.server` is unaffected by any of this.
"""

import asyncio
import hmac
import json
import logging
import re
from contextlib import asynccontextmanager, suppress

from fastapi import FastAPI
from mcp.server import Server, ServerRequestContext
from mcp.server.sse import SseServerTransport
from mcp.server.transport_security import TransportSecurityMiddleware, TransportSecuritySettings
from mcp.types import CallToolRequestParams, CallToolResult, ListToolsResult, PaginatedRequestParams
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import Response
from starlette.types import ASGIApp, Receive, Scope, Send

from sktime_mcp import __version__
from sktime_mcp.config import settings
from sktime_mcp.server import _error_result, _periodic_job_cleanup, call_tool, list_tools

logger = logging.getLogger(__name__)

HEALTH_PATH = "/"


class BearerTokenMiddleware:
    """Pure ASGI middleware: reject any HTTP request without the shared bearer token.

    ``BaseHTTPMiddleware`` is avoided on purpose; it buffers streaming responses,
    which breaks the SSE channel.
    """

    def __init__(self, app: ASGIApp, token: str, exempt_paths: frozenset[str]) -> None:
        self.app = app
        self._token = token.encode()
        self._exempt_paths = exempt_paths

    def _is_authorized(self, scope: Scope) -> bool:
        header = next(
            (value for key, value in scope["headers"] if key == b"authorization"),
            b"",
        )
        scheme, _, presented = header.partition(b" ")
        if scheme.lower() != b"bearer":
            return False
        return hmac.compare_digest(presented.strip(), self._token)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope["path"] in self._exempt_paths:
            await self.app(scope, receive, send)
            return
        if not self._is_authorized(scope):
            logger.warning("Rejected unauthenticated %s %s", scope["method"], scope["path"])
            response = Response(
                json.dumps({"error": "unauthorized", "detail": "Bearer token missing or invalid"}),
                status_code=401,
                media_type="application/json",
                headers={"WWW-Authenticate": "Bearer"},
            )
            await response(scope, receive, send)
            return
        await self.app(scope, receive, send)


def _origin_regex(allowed_origins: list[str]) -> str:
    """Translate the mcp-style origin allow-list (``scheme://host:*`` = any port) to a regex.

    ``CORSMiddleware.allow_origins`` is exact-match only, so the wildcard-port
    entries need ``allow_origin_regex``.
    """
    alternatives = []
    for origin in allowed_origins:
        if origin.endswith(":*"):
            alternatives.append(re.escape(origin[:-2]) + r":\d+")
        else:
            alternatives.append(re.escape(origin))
    return "^(?:" + "|".join(alternatives) + ")$"


def _http_only_server(hidden_tools: frozenset[str]) -> Server:
    """A ``Server`` sharing the stdio server's handlers, minus ``hidden_tools``."""

    async def http_list_tools(
        ctx: ServerRequestContext, params: PaginatedRequestParams | None
    ) -> ListToolsResult:
        result = await list_tools(ctx, params)
        return ListToolsResult(tools=[t for t in result.tools if t.name not in hidden_tools])

    async def http_call_tool(
        ctx: ServerRequestContext, params: CallToolRequestParams
    ) -> CallToolResult:
        if params.name in hidden_tools:
            return _error_result(
                f"Tool '{params.name}' is disabled over HTTP/SSE "
                "(set SKTIME_MCP_HTTP_DISABLE_RUN_COMMAND=false to enable it)."
            )
        return await call_tool(ctx, params)

    return Server(
        "sktime-mcp",
        version=__version__,
        on_list_tools=http_list_tools,
        on_call_tool=http_call_tool,
    )


def create_app() -> FastAPI:
    """Build the FastAPI app from the ``SKTIME_MCP_HTTP_*`` settings.

    Raises ``RuntimeError`` when no token is configured and
    ``SKTIME_MCP_HTTP_INSECURE`` is not ``true``.
    """
    token = settings.http_token
    if token is None and not settings.http_insecure:
        raise RuntimeError(
            "sktime-mcp HTTP/SSE refuses to start without authentication: set "
            "SKTIME_MCP_HTTP_TOKEN to a secret bearer token, or set "
            "SKTIME_MCP_HTTP_INSECURE=true to serve without one (local, isolated networks only)."
        )
    if token is None:
        logger.warning(
            "SKTIME_MCP_HTTP_INSECURE=true: the HTTP/SSE app is serving WITHOUT authentication. "
            "Anyone who can reach this port can drive the server as your user. "
            "Do not expose it beyond localhost."
        )

    allowed_hosts = settings.http_allowed_hosts
    allowed_origins = settings.http_allowed_origins
    security = TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=allowed_hosts,
        allowed_origins=allowed_origins,
    )
    # Validates GET /sse before the stream is opened; the transport re-checks
    # inside connect_sse, but by then a failure surfaces as a raised ValueError.
    host_check = TransportSecurityMiddleware(security)
    sse = SseServerTransport("/messages/", security_settings=security)

    hidden_tools = frozenset({"run_command"}) if settings.http_disable_run_command else frozenset()
    mcp_server = _http_only_server(hidden_tools)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        cleanup_task = asyncio.create_task(_periodic_job_cleanup())
        logger.info("sktime-mcp FastAPI server starting up...")
        yield
        cleanup_task.cancel()
        with suppress(asyncio.CancelledError):
            await cleanup_task
        logger.info("sktime-mcp FastAPI server shut down.")

    app = FastAPI(
        title="sktime-mcp",
        description="MCP (Model Context Protocol) layer for sktime, accessible via HTTP/SSE.",
        version="0.1.0",
        lifespan=lifespan,
    )

    @app.get("/sse")
    async def handle_sse(request: Request):
        """Open the SSE channel; the MCP session runs for as long as it stays open."""
        rejected = await host_check.validate_request(request, is_post=False)
        if rejected is not None:
            return rejected
        async with sse.connect_sse(request.scope, request.receive, request._send) as streams:
            await mcp_server.run(streams[0], streams[1], mcp_server.create_initialization_options())
        # The transport has already written the whole response; return an empty
        # one so the route does not try to send a second (None) response.
        return Response()

    # The transport answers the POST itself (202 Accepted / 4xx), so it is
    # mounted as a raw ASGI app; wrapping it in a route would send a second response.
    app.mount("/messages/", app=sse.handle_post_message)

    @app.get(HEALTH_PATH)
    def read_root():
        """Health check (the only unauthenticated endpoint)."""
        return {
            "status": "online",
            "service": "sktime-mcp",
            "endpoints": {"sse": "/sse", "messages": "/messages/"},
            "docs": "/docs",
        }

    # Innermost first: auth runs after CORS, so a preflight (which carries no
    # Authorization header) is answered by CORSMiddleware and never reaches it.
    if token is not None:
        app.add_middleware(
            BearerTokenMiddleware, token=token, exempt_paths=frozenset({HEALTH_PATH})
        )
    app.add_middleware(
        CORSMiddleware,
        allow_origin_regex=_origin_regex(allowed_origins),
        allow_methods=["GET", "POST"],
        allow_headers=["Authorization", "Content-Type"],
    )
    return app


app = create_app()
