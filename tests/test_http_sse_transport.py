"""End-to-end test for the HTTP/SSE transport (issue #556 item 2).

Spins up ``uvicorn sktime_mcp.app:app`` as a subprocess and drives it with
the official MCP client (``mcp.client.sse.sse_client``).

Background: the SSE handlers used to be FastAPI routes that drove the MCP
transport over ``request._send`` and then returned ``None``. FastAPI emitted
a second response after the transport had already completed the first one,
so uvicorn tore the connection down with ``RuntimeError: Unexpected ASGI
message 'http.response.start' sent, after response already completed`` and
``initialize()`` never completed. The app now wires the transport per the
MCP SDK's SSE server pattern (a plain Starlette ``Route`` returning an
empty ``Response`` for ``/sse`` and a ``Mount`` for ``/messages/``).
"""

import asyncio
import os
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, "src")

from mcp import ClientSession
from mcp.client.sse import sse_client

REPO_ROOT = Path(__file__).resolve().parent.parent

SERVER_START_TIMEOUT = 90.0
CLIENT_TIMEOUT = 30.0


def _local_client_factory(headers=None, timeout=None, auth=None):
    """Build an httpx client that ignores proxy env vars.

    The test server listens on 127.0.0.1, which must never go through a
    proxy; ignoring the environment also keeps the test hermetic on CI
    machines with proxies configured.
    """
    return httpx.AsyncClient(timeout=timeout, trust_env=False)


def _free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _wait_for_port(port):
    deadline = time.time() + SERVER_START_TIMEOUT
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=2):
                return
        except OSError:
            time.sleep(0.5)
    raise TimeoutError(f"uvicorn did not start on 127.0.0.1:{port} in time")


@pytest.fixture()
def sse_server():
    """Start ``uvicorn sktime_mcp.app:app``; yield ``(base_url, log_path)``."""
    port = _free_port()
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".log", prefix="sktime-mcp-sse-", delete=False
    ) as log_file:
        log_path = log_file.name
    env = dict(os.environ, PYTHONPATH=str(REPO_ROOT / "src"))
    with Path(log_path).open("w") as log_sink:
        proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "sktime_mcp.app:app",
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
            ],
            cwd=REPO_ROOT,
            env=env,
            stdout=log_sink,
            stderr=subprocess.STDOUT,
        )
    try:
        _wait_for_port(port)
        yield f"http://127.0.0.1:{port}", log_path
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=30)


async def test_sse_initialize_and_list_tools(sse_server):
    """initialize() and list_tools() complete over SSE with no ASGI errors."""
    base_url, log_path = sse_server
    async with (
        sse_client(f"{base_url}/sse", httpx_client_factory=_local_client_factory) as (
            read,
            write,
        ),
        ClientSession(read, write) as session,
    ):
        init = await asyncio.wait_for(session.initialize(), timeout=CLIENT_TIMEOUT)
        assert init.serverInfo.name == "sktime-mcp"
        tools = await asyncio.wait_for(session.list_tools(), timeout=CLIENT_TIMEOUT)
        assert len(tools.tools) > 0
    # Give the server a beat to flush its logs, then confirm the transport
    # never double-sent a response (the #556 item 2 failure mode).
    await asyncio.sleep(2)
    server_log = Path(log_path).read_text()
    assert "Unexpected ASGI message" not in server_log
