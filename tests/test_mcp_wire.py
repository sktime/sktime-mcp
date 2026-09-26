"""Wire-level tests: drive the server through the mcp client over real transports.

The unit tests call the tool functions directly, so they never exercise the
MCP layer itself (handler wiring, ``inputSchema`` serialization, argument
validation, the ``isError`` flag, stdio hygiene). These tests do.
"""

import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from contextlib import asynccontextmanager
from pathlib import Path

import anyio
import pytest
from mcp.client.session import ClientSession
from mcp.client.sse import sse_client
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.shared.memory import create_client_server_memory_streams

import sktime_mcp
from sktime_mcp.server import get_tools, server

EXPECTED_TOOL_COUNT = 26


@asynccontextmanager
async def _connected_session():
    """Run ``server`` in-process and yield an initialized client session."""
    async with (
        create_client_server_memory_streams() as (client_streams, server_streams),
        anyio.create_task_group() as tg,
    ):
        tg.start_soon(
            server.run,
            server_streams[0],
            server_streams[1],
            server.create_initialization_options(),
        )
        async with ClientSession(*client_streams) as session:
            await session.initialize()
            yield session
        tg.cancel_scope.cancel()


def _body(result) -> dict:
    assert len(result.content) == 1
    assert result.content[0].type == "text"
    return json.loads(result.content[0].text)


async def test_initialize_reports_server_identity():
    async with _connected_session() as session:
        init = await session.initialize()
    assert init.server_info.name == "sktime-mcp"
    assert init.server_info.version == sktime_mcp.__version__
    assert init.capabilities.tools is not None


async def test_list_tools_over_the_wire():
    async with _connected_session() as session:
        listed = await session.list_tools()

    assert len(listed.tools) == EXPECTED_TOOL_COUNT
    assert {t.name for t in listed.tools} == {t.name for t in get_tools()}
    for tool in listed.tools:
        # ``input_schema`` is populated from the ``inputSchema`` key of the
        # serialized message, so a non-empty schema here means it was on the wire.
        assert tool.input_schema, f"{tool.name} has no inputSchema on the wire"
        assert tool.input_schema["type"] == "object"
        assert "inputSchema" in tool.model_dump(by_alias=True, exclude_unset=True)


async def test_call_tool_returns_structured_success():
    async with _connected_session() as session:
        result = await session.call_tool("query_registry", {"task": "forecaster", "limit": 1})

    assert result.is_error is False
    body = _body(result)
    assert body["success"] is True
    assert len(body["results"]) == 1


async def test_wrongly_typed_argument_is_rejected_before_dispatch(monkeypatch):
    def _must_not_run(*args, **kwargs):
        raise AssertionError("tool code ran despite the invalid argument")

    monkeypatch.setattr("sktime_mcp.server.list_available_data_tool", _must_not_run)

    async with _connected_session() as session:
        result = await session.call_tool("list_available_data", {"is_demo": "false"})

    assert result.is_error is True
    body = _body(result)
    assert body["success"] is False
    assert body["error"].startswith("Input validation error:")
    assert "'false' is not of type 'boolean'" in body["error"]


async def test_tool_exception_becomes_structured_error(monkeypatch):
    def _boom(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr("sktime_mcp.server.query_registry_tool", _boom)

    async with _connected_session() as session:
        result = await session.call_tool("query_registry", {"task": "forecaster"})

    assert result.is_error is True
    assert _body(result) == {"success": False, "error": "boom"}


_NOISY_SERVER = """
import os, sys
import sktime_mcp.server as srv

_real = srv.query_registry_tool

def noisy(**kwargs):
    print("STRAY PYTHON PRINT")
    os.write(1, b"STRAY FD-LEVEL WRITE\\n")
    return _real(**kwargs)

srv.query_registry_tool = noisy
srv.main()
"""


@pytest.mark.skipif(sys.platform == "win32", reason="fd-level stdio test is POSIX-only")
async def test_stray_stdout_output_does_not_corrupt_stdio_protocol(tmp_path: Path):
    src_dir = str(Path(sktime_mcp.__file__).resolve().parents[1])
    env = {**os.environ, "PYTHONPATH": src_dir}
    params = StdioServerParameters(command=sys.executable, args=["-c", _NOISY_SERVER], env=env)
    stderr_path = tmp_path / "server-stderr.txt"

    with stderr_path.open("w") as errlog:
        async with stdio_client(params, errlog=errlog) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(
                    "query_registry", {"task": "forecaster", "limit": 1}
                )

    assert result.is_error is False
    assert _body(result)["success"] is True

    stderr = stderr_path.read_text()
    assert "STRAY PYTHON PRINT" in stderr
    assert "STRAY FD-LEVEL WRITE" in stderr


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _wait_for_http(url: str, proc: subprocess.Popen, timeout: float = 90.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"uvicorn exited early with code {proc.returncode}")
        try:
            with urllib.request.urlopen(url, timeout=1):
                return
        except (urllib.error.URLError, OSError):
            time.sleep(0.25)
    raise TimeoutError(f"{url} did not come up within {timeout}s")


@pytest.mark.skipif(sys.platform == "win32", reason="uses POSIX process handling")
async def test_sse_app_completes_initialize_and_serves_tools(tmp_path: Path):
    """The HTTP/SSE app must complete the MCP handshake (regression for audit F-02).

    Before the ``/messages/`` endpoint was mounted as a raw ASGI app, FastAPI
    sent a second response after the transport's 202, so initialize never
    completed.
    """
    src_dir = str(Path(sktime_mcp.__file__).resolve().parents[1])
    port = _free_port()
    log_path = tmp_path / "uvicorn.log"
    with log_path.open("w") as log:
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
            env={**os.environ, "PYTHONPATH": src_dir},
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        try:
            _wait_for_http(f"http://127.0.0.1:{port}/", proc)
            with anyio.fail_after(60):
                async with sse_client(f"http://127.0.0.1:{port}/sse") as (read, write):
                    async with ClientSession(read, write) as session:
                        init = await session.initialize()
                        listed = await session.list_tools()
                        result = await session.call_tool(
                            "query_registry", {"task": "forecaster", "limit": 1}
                        )
        finally:
            proc.terminate()
            proc.wait(timeout=15)

    assert init.server_info.name == "sktime-mcp"
    assert len(listed.tools) == EXPECTED_TOOL_COUNT
    assert result.is_error is False
    assert _body(result)["success"] is True
    assert "Unexpected ASGI message" not in log_path.read_text()
