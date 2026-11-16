import asyncio
import hmac
import logging
from contextlib import asynccontextmanager, suppress

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from mcp.server.sse import SseServerTransport
from mcp.server.transport_security import TransportSecuritySettings
from starlette.datastructures import Headers
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.types import ASGIApp, Receive, Scope, Send

from sktime_mcp.config import settings
from sktime_mcp.server import _periodic_job_cleanup, server

logger = logging.getLogger(__name__)

# Paths reachable without a token (health check only).
PUBLIC_PATHS = frozenset({"/"})


class BearerTokenMiddleware:
    """Reject HTTP requests that do not carry ``Authorization: Bearer <token>``.

    Plain ASGI middleware (not ``BaseHTTPMiddleware``) so the SSE stream is
    passed through untouched.
    """

    def __init__(self, app: ASGIApp, token: str | None):
        self.app = app
        self.token = token

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if self.token is None or scope["type"] != "http" or scope["path"] in PUBLIC_PATHS:
            await self.app(scope, receive, send)
            return

        scheme, _, value = Headers(scope=scope).get("authorization", "").partition(" ")
        if scheme.lower() == "bearer" and hmac.compare_digest(value.encode(), self.token.encode()):
            await self.app(scope, receive, send)
            return

        response = JSONResponse(
            {"error": "Missing or invalid bearer token"},
            status_code=401,
            headers={"WWW-Authenticate": "Bearer"},
        )
        await response(scope, receive, send)


# Define SSE transport on a relative path. The transport checks Host/Origin
# headers itself to block DNS-rebinding attacks.
sse = SseServerTransport(
    "/messages/",
    security_settings=TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=settings.http_allowed_hosts,
        allowed_origins=settings.http_allowed_origins,
    ),
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle events for the FastAPI app."""
    if settings.http_token is None:
        if not settings.http_insecure:
            raise RuntimeError(
                "Refusing to start the HTTP server without authentication. "
                "Set SKTIME_MCP_HTTP_TOKEN, or set SKTIME_MCP_HTTP_INSECURE=true "
                "to run without a token (local use only)."
            )
        logger.warning("SKTIME_MCP_HTTP_INSECURE=true: HTTP server has no authentication.")

    # Start the periodic job cleanup background task
    cleanup_task = asyncio.create_task(_periodic_job_cleanup())
    logger.info("sktime-mcp FastAPI server starting up...")

    yield

    # Cleanup on shutdown
    cleanup_task.cancel()
    with suppress(asyncio.CancelledError):
        await cleanup_task
    logger.info("sktime-mcp FastAPI server shut down.")


# Initialize the FastAPI app
app = FastAPI(
    title="sktime-mcp",
    description="MCP (Model Context Protocol) layer for sktime, accessible via HTTP/SSE.",
    version="0.1.0",
    lifespan=lifespan,
)

# Added last so CORS is the outermost layer and answers preflight requests,
# which never carry the Authorization header.
app.add_middleware(BearerTokenMiddleware, token=settings.http_token)
if settings.http_allowed_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.http_allowed_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["Authorization", "Content-Type"],
    )


@app.get("/sse")
async def handle_sse(request: Request):
    """
    Endpoint to initiate an SSE connection.
    Clients should connect here to receive server messages.
    """
    async with sse.connect_sse(
        request.scope,
        request.receive,
        request._send,
    ) as streams:
        # Run the MCP server over the established memory streams
        await server.run(
            streams[0],
            streams[1],
            server.create_initialization_options(),
        )
    # The SSE transport has already written the whole HTTP response; return an
    # empty one so the route does not try to send a second (None) response.
    return Response()


# The transport answers the POST itself (202 Accepted / 4xx), so it is mounted
# as a raw ASGI app; wrapping it in a route would send a second response.
app.mount("/messages/", app=sse.handle_post_message)


@app.get("/")
def read_root():
    """Health check and simple landing page."""
    return {
        "status": "online",
        "service": "sktime-mcp",
        "endpoints": {"sse": "/sse", "messages": "/messages/"},
        "docs": "/docs",
    }
