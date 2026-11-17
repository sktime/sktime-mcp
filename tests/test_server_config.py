"""Tests for server environment configuration parsing."""

import importlib
import sys

sys.path.insert(0, "src")


def _reload_server_module():
    """Reload server module so import-time env parsing is re-evaluated."""
    if "sktime_mcp.server" in sys.modules:
        return importlib.reload(sys.modules["sktime_mcp.server"])

    return importlib.import_module("sktime_mcp.server")


def test_invalid_job_max_age_env_falls_back_to_default(monkeypatch):
    """Invalid max-age env values should not crash server import."""
    monkeypatch.setenv("SKTIME_MCP_JOB_MAX_AGE_HOURS", "abc")
    monkeypatch.delenv("SKTIME_MCP_JOB_CLEANUP_INTERVAL", raising=False)

    server = _reload_server_module()

    assert server.JOB_MAX_AGE_HOURS == 24


def test_invalid_job_cleanup_interval_env_falls_back_to_default(monkeypatch):
    """Invalid cleanup-interval env values should use the default."""
    monkeypatch.setenv("SKTIME_MCP_JOB_CLEANUP_INTERVAL", "abc")
    monkeypatch.delenv("SKTIME_MCP_JOB_MAX_AGE_HOURS", raising=False)

    server = _reload_server_module()

    assert server.JOB_CLEANUP_INTERVAL_SECS == 3600


def test_valid_numeric_server_env_values_are_respected(monkeypatch):
    """Valid numeric env values should still override defaults."""
    monkeypatch.setenv("SKTIME_MCP_JOB_MAX_AGE_HOURS", "48")
    monkeypatch.setenv("SKTIME_MCP_JOB_CLEANUP_INTERVAL", "120")

    server = _reload_server_module()

    assert server.JOB_MAX_AGE_HOURS == 48
    assert server.JOB_CLEANUP_INTERVAL_SECS == 120
