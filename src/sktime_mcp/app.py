import asyncio
import logging
from contextlib import asynccontextmanager, suppress

from fastapi import FastAPI
from mcp.server.sse import SseServerTransport
from starlette.requests import Request

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


# Initialize the FastAPI app
app = FastAPI(
    title="sktime-mcp",
    description="MCP (Model Context Protocol) layer for sktime, accessible via HTTP/SSE.",
    version="0.1.0",
    lifespan=lifespan,
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


@app.post("/messages/")
async def handle_messages(request: Request):
    """
    Endpoint for the client to POST JSON-RPC messages.
    The client must include the ?session_id=... query parameter.
    """
    await sse.handle_post_message(
        request.scope,
        request.receive,
        request._send,
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
