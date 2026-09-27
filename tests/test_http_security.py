"""HTTP/SSE hardening (audit F-03 / F-41, #556 item 3).

The app used to serve ``run_command`` to anyone who could reach the port, with
no authentication, no Host/Origin checks and no CORS policy. These tests pin
the bearer-token gate, DNS-rebinding protection, CORS and the HTTP-only
``run_command`` switch. Pure request/response behaviour goes through
``TestClient``; the token end-to-end path uses a real uvicorn subprocess.
"""

import json
import logging
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import anyio
import pytest
from mcp.client.session import ClientSession
from mcp.client.sse import sse_client
from mcp.shared.memory import create_client_server_memory_streams
from starlette.testclient import TestClient

import sktime_mcp

TOKEN = "test-token-f03"
AUTH = {"Authorization": f"Bearer {TOKEN}"}


@pytest.fixture
def app_module(monkeypatch):
    """Import ``sktime_mcp.app`` with a token set (its module-level app needs one)."""
    monkeypatch.setenv("SKTIME_MCP_HTTP_TOKEN", TOKEN)
    monkeypatch.delenv("SKTIME_MCP_HTTP_INSECURE", raising=False)
    monkeypatch.delenv("SKTIME_MCP_HTTP_ALLOWED_HOSTS", raising=False)
    monkeypatch.delenv("SKTIME_MCP_HTTP_ALLOWED_ORIGINS", raising=False)
    monkeypatch.delenv("SKTIME_MCP_HTTP_DISABLE_RUN_COMMAND", raising=False)
    import sktime_mcp.app as app_module

    return app_module


@pytest.fixture
def client(app_module):
    # ``base_url`` sets the Host header; the default ``testserver`` is not an allowed host.
    return TestClient(app_module.create_app(), base_url="http://localhost")


# -- bearer token ----------------------------------------------------------


def test_health_check_stays_public(client):
    assert client.get("/").status_code == 200


@pytest.mark.parametrize("path", ["/sse", "/messages/?session_id=abc"])
def test_missing_token_is_401_with_json_body(client, path):
    response = client.post(path, json={}) if "messages" in path else client.get(path)

    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"
    assert response.json()["error"] == "unauthorized"


@pytest.mark.parametrize(
    "header",
    [
        {"Authorization": "Bearer wrong-token"},
        {"Authorization": "Bearer "},
        {"Authorization": f"Basic {TOKEN}"},
        {"Authorization": f"Bearer {TOKEN}x"},
    ],
)
def test_wrong_token_is_401(client, header):
    response = client.post("/messages/?session_id=abc", json={}, headers=header)

    assert response.status_code == 401
    assert response.json()["error"] == "unauthorized"


def test_right_token_reaches_the_transport(client):
    # 400 "Invalid session ID" comes from the SSE transport, i.e. auth passed.
    response = client.post("/messages/?session_id=abc", json={}, headers=AUTH)

    assert response.status_code == 400
    assert "session" in response.text.lower()


def test_missing_token_env_refuses_to_build_the_app(app_module, monkeypatch):
    monkeypatch.delenv("SKTIME_MCP_HTTP_TOKEN")

    with pytest.raises(RuntimeError, match="SKTIME_MCP_HTTP_TOKEN"):
        app_module.create_app()


def test_blank_token_env_counts_as_unset(app_module, monkeypatch):
    monkeypatch.setenv("SKTIME_MCP_HTTP_TOKEN", "   ")

    with pytest.raises(RuntimeError, match="SKTIME_MCP_HTTP_TOKEN"):
        app_module.create_app()


def test_insecure_flag_serves_without_token_and_warns(app_module, monkeypatch, caplog):
    monkeypatch.delenv("SKTIME_MCP_HTTP_TOKEN")
    monkeypatch.setenv("SKTIME_MCP_HTTP_INSECURE", "true")

    with caplog.at_level(logging.WARNING, logger="sktime_mcp.app"):
        app = app_module.create_app()

    assert any(
        "WITHOUT authentication" in record.getMessage() and record.levelno == logging.WARNING
        for record in caplog.records
    )
    response = TestClient(app, base_url="http://localhost").post(
        "/messages/?session_id=abc", json={}
    )
    assert response.status_code == 400  # transport reached, not 401


# -- DNS-rebinding protection (Host / Origin) --------------------------------


def test_bad_host_header_is_rejected_on_sse(client):
    response = client.get("/sse", headers={**AUTH, "Host": "evil.example.com"})

    assert response.status_code == 421


def test_bad_host_header_is_rejected_on_messages(client):
    response = client.post(
        "/messages/?session_id=abc", json={}, headers={**AUTH, "Host": "evil.example.com"}
    )

    assert response.status_code == 421


def test_bad_origin_header_is_rejected(client):
    response = client.get("/sse", headers={**AUTH, "Origin": "http://evil.example.com"})

    assert response.status_code == 403


def test_allowed_hosts_env_extends_the_allow_list(app_module, monkeypatch):
    monkeypatch.setenv("SKTIME_MCP_HTTP_ALLOWED_HOSTS", "mcp.internal:*, localhost")
    client = TestClient(app_module.create_app(), base_url="http://mcp.internal:8001")

    response = client.post("/messages/?session_id=abc", json={}, headers=AUTH)

    assert response.status_code == 400  # past the Host check


# -- CORS ------------------------------------------------------------------


def _preflight(client, origin):
    return client.options(
        "/messages/",
        headers={
            "Origin": origin,
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "authorization, content-type",
        },
    )


@pytest.mark.parametrize(
    "origin", ["http://localhost:5173", "http://127.0.0.1", "http://localhost"]
)
def test_cors_preflight_from_allowed_origin_gets_the_headers(client, origin):
    response = _preflight(client, origin)

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == origin
    allowed_headers = response.headers["access-control-allow-headers"].lower()
    assert "authorization" in allowed_headers


@pytest.mark.parametrize("origin", ["http://evil.example.com", "http://localhost.evil.com:80"])
def test_cors_preflight_from_disallowed_origin_gets_no_header(client, origin):
    response = _preflight(client, origin)

    assert "access-control-allow-origin" not in response.headers


def test_allowed_origins_env_is_used_for_cors(app_module, monkeypatch):
    monkeypatch.setenv("SKTIME_MCP_HTTP_ALLOWED_ORIGINS", "https://app.example.com")
    client = TestClient(app_module.create_app(), base_url="http://localhost")

    assert (
        _preflight(client, "https://app.example.com").headers["access-control-allow-origin"]
        == "https://app.example.com"
    )
    assert "access-control-allow-origin" not in _preflight(client, "http://localhost").headers


# -- run_command over HTTP ---------------------------------------------------


def _body(result) -> dict:
    assert len(result.content) == 1
    return json.loads(result.content[0].text)


async def _list_and_call_run_command(mcp_server):
    async with (
        create_client_server_memory_streams() as (client_streams, server_streams),
        anyio.create_task_group() as tg,
    ):
        tg.start_soon(
            mcp_server.run,
            server_streams[0],
            server_streams[1],
            mcp_server.create_initialization_options(),
        )
        async with ClientSession(*client_streams) as session:
            await session.initialize()
            listed = await session.list_tools()
            called = await session.call_tool("run_command", {"command": "echo should-not-run"})
        tg.cancel_scope.cancel()
    return listed, called


async def test_http_server_hides_and_rejects_run_command(app_module):
    from sktime_mcp.server import get_tools

    http_server = app_module._http_only_server(frozenset({"run_command"}))

    listed, called = await _list_and_call_run_command(http_server)

    names = {t.name for t in listed.tools}
    assert "run_command" not in names
    assert names == {t.name for t in get_tools()} - {"run_command"}
    assert called.is_error is True
    body = _body(called)
    assert body["success"] is False
    assert "disabled over HTTP" in body["error"]
    assert "SKTIME_MCP_HTTP_DISABLE_RUN_COMMAND" in body["error"]


async def test_run_command_switch_can_be_turned_off(app_module):
    http_server = app_module._http_only_server(frozenset())

    listed, called = await _list_and_call_run_command(http_server)

    assert "run_command" in {t.name for t in listed.tools}
    assert called.is_error is False
    assert "should-not-run" in _body(called)["output"]


async def test_stdio_server_still_exposes_run_command():
    from sktime_mcp.server import server

    listed, _ = await _list_and_call_run_command(server)

    assert "run_command" in {t.name for t in listed.tools}


# -- end to end over a real uvicorn process ------------------------------------


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
async def test_token_end_to_end_over_uvicorn(tmp_path: Path):
    src_dir = str(Path(sktime_mcp.__file__).resolve().parents[1])
    port = _free_port()
    base = f"http://127.0.0.1:{port}"
    env = {**os.environ, "PYTHONPATH": src_dir, "SKTIME_MCP_HTTP_TOKEN": TOKEN}
    env.pop("SKTIME_MCP_HTTP_DISABLE_RUN_COMMAND", None)
    log_path = tmp_path / "uvicorn.log"
    with log_path.open("w") as log:
        proc = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "sktime_mcp.app:app", "--port", str(port)],
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        try:
            _wait_for_http(f"{base}/", proc)

            with pytest.raises(urllib.error.HTTPError) as exc_info:
                urllib.request.urlopen(f"{base}/sse", timeout=5)
            assert exc_info.value.code == 401

            with anyio.fail_after(60):
                async with sse_client(f"{base}/sse", headers=AUTH) as (read, write):
                    async with ClientSession(read, write) as session:
                        init = await session.initialize()
                        listed = await session.list_tools()
                        rejected = await session.call_tool("run_command", {"command": "id"})
        finally:
            proc.terminate()
            proc.wait(timeout=15)

    assert init.server_info.name == "sktime-mcp"
    assert "run_command" not in {t.name for t in listed.tools}
    assert rejected.is_error is True
    assert "disabled over HTTP" in _body(rejected)["error"]
    assert "Unexpected ASGI message" not in log_path.read_text()
