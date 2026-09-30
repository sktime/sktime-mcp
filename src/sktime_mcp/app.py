import asyncio
import logging
from contextlib import asynccontextmanager, suppress

from fastapi import FastAPI
from mcp.server.sse import SseServerTransport
from starlette.requests import Request
from starlette.responses import Response
from starlette.routing import Mount, Route

from sktime_mcp.server import _periodic_job_cleanup, server

logger = logging.getLogger(__name__)

# Define SSE transport on a relative path
sse = SseServerTransport("/messages/")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle events for the FastAPI app."""
    # Start the periodic job cleanup background task
    cleanup_task = asyncio.create_task(_periodic_job_cleanup())
    logger.info("sktime-mcp FastAPI server starting up...")

    yield

    # Cleanup on shutdown
    cleanup_task.cancel()
    with suppress(asyncio.CancelledError):
        await cleanup_task
    logger.info("sktime-mcp FastAPI server shut down.")


async def handle_sse(request: Request) -> Response:
    """Initiate an SSE connection.

    Clients should connect here to receive server messages.

    The transport drives the response itself over the raw ASGI channel, so
    this handler must return a plain ``Response`` (never ``None``): a plain
    Starlette route simply sends back whatever is returned, while a FastAPI
    route would emit a second response after the transport already completed
    the first one.
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
    # The SSE stream has ended; hand the route wrapper an empty response so
    # it has something to send (per the MCP SDK's SSE server example).
    return Response()


# Initialize the FastAPI app.
# The SSE and message endpoints are plain Starlette routes, not FastAPI
# routes: the MCP transport speaks raw ASGI and completes each response
# itself, so FastAPI's response machinery must stay out of the way.
app = FastAPI(
    title="sktime-mcp",
    description="MCP (Model Context Protocol) layer for sktime, accessible via HTTP/SSE.",
    version="0.1.0",
    lifespan=lifespan,
    routes=[
        Route("/sse", endpoint=handle_sse, methods=["GET"]),
        Mount("/messages/", app=sse.handle_post_message),
    ],
)


@app.get("/")
def read_root():
    """Health check and simple landing page."""
    return {
        "status": "online",
        "service": "sktime-mcp",
        "endpoints": {"sse": "/sse", "messages": "/messages/"},
        "docs": "/docs",
    }
