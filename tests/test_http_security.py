"""HTTP app auth, DNS-rebinding and CORS settings (#556, item 3)."""

import importlib

import pytest
from starlette.testclient import TestClient

TOKEN = "s3cret-token"
HTTP_ENV = (
    "SKTIME_MCP_HTTP_TOKEN",
    "SKTIME_MCP_HTTP_INSECURE",
    "SKTIME_MCP_HTTP_ALLOWED_HOSTS",
    "SKTIME_MCP_HTTP_ALLOWED_ORIGINS",
)


@pytest.fixture
def make_client(monkeypatch):
    """Reload ``sktime_mcp.app`` under the given env and return a test client."""

    def _make(**env):
        for name in HTTP_ENV:
            monkeypatch.delenv(name, raising=False)
        for name, value in env.items():
            monkeypatch.setenv(name, value)
        import sktime_mcp.app

        app_module = importlib.reload(sktime_mcp.app)
        return TestClient(app_module.app, base_url="http://localhost:8001")

    return _make


def _post_message(client, **headers):
    # No session_id: a request that gets past auth and host checks is answered
    # by the MCP transport itself with 400 "session_id is required".
    return client.post(
        "/messages/",
        content="{}",
        headers={"content-type": "application/json", **headers},
    )


def test_refuses_to_start_without_token(make_client):
    client = make_client()
    with pytest.raises(RuntimeError, match="SKTIME_MCP_HTTP_TOKEN"), client:
        pass


def test_insecure_starts_without_token(make_client):
    client = make_client(SKTIME_MCP_HTTP_INSECURE="true")
    with client:
        response = _post_message(client)
    assert response.status_code == 400


def test_missing_or_wrong_token_is_rejected(make_client):
    client = make_client(SKTIME_MCP_HTTP_TOKEN=TOKEN)
    with client:
        missing = _post_message(client)
        wrong = _post_message(client, authorization="Bearer nope")

    assert missing.status_code == 401
    assert missing.headers["www-authenticate"] == "Bearer"
    assert wrong.status_code == 401


def test_health_check_needs_no_token(make_client):
    client = make_client(SKTIME_MCP_HTTP_TOKEN=TOKEN)
    with client:
        response = client.get("/")
    assert response.status_code == 200


def test_unknown_host_is_rejected(make_client):
    client = make_client(SKTIME_MCP_HTTP_TOKEN=TOKEN)
    with client:
        response = _post_message(client, authorization=f"Bearer {TOKEN}", host="evil.example.com")
    assert response.status_code == 421


def test_allowed_hosts_are_configurable(make_client):
    client = make_client(
        SKTIME_MCP_HTTP_TOKEN=TOKEN,
        SKTIME_MCP_HTTP_ALLOWED_HOSTS="abc.ngrok-free.app",
    )
    with client:
        response = _post_message(client, authorization=f"Bearer {TOKEN}", host="abc.ngrok-free.app")
    assert response.status_code == 400


def test_only_allowed_origins_pass(make_client):
    origin = "https://chat.example.com"
    client = make_client(
        SKTIME_MCP_HTTP_TOKEN=TOKEN,
        SKTIME_MCP_HTTP_ALLOWED_ORIGINS=origin,
    )
    with client:
        allowed = _post_message(client, authorization=f"Bearer {TOKEN}", origin=origin)
        other = _post_message(
            client, authorization=f"Bearer {TOKEN}", origin="https://evil.example.com"
        )
    assert allowed.status_code == 400
    assert other.status_code == 403


def test_cors_preflight_only_for_allowed_origins(make_client):
    origin = "https://chat.example.com"
    client = make_client(
        SKTIME_MCP_HTTP_TOKEN=TOKEN,
        SKTIME_MCP_HTTP_ALLOWED_ORIGINS=origin,
    )
    preflight = {"access-control-request-method": "POST"}
    with client:
        allowed = client.options("/messages/", headers={"origin": origin, **preflight})
        other = client.options(
            "/messages/", headers={"origin": "https://evil.example.com", **preflight}
        )

    assert allowed.status_code == 200
    assert allowed.headers["access-control-allow-origin"] == origin
    assert "access-control-allow-origin" not in other.headers
