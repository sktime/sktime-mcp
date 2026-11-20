"""Tests for server environment configuration parsing.

The job settings live on ``sktime_mcp.config.settings`` and are read from the
environment on every access, so no module reload is needed. The detailed
parsing rules (warnings, minimums, every integer setting) are covered in
``tests/test_env_config.py``; this file keeps the original server-level
regression cases.
"""

from sktime_mcp.config import settings


def test_invalid_job_max_age_env_falls_back_to_default(monkeypatch):
    """Invalid max-age env values should fall back rather than raise."""
    monkeypatch.setenv("SKTIME_MCP_JOB_MAX_AGE_HOURS", "abc")
    monkeypatch.delenv("SKTIME_MCP_JOB_CLEANUP_INTERVAL", raising=False)

    assert settings.job_max_age_hours == 24


def test_invalid_job_cleanup_interval_env_falls_back_to_default(monkeypatch):
    """Invalid cleanup-interval env values should use the default."""
    monkeypatch.setenv("SKTIME_MCP_JOB_CLEANUP_INTERVAL", "abc")
    monkeypatch.delenv("SKTIME_MCP_JOB_MAX_AGE_HOURS", raising=False)

    assert settings.job_cleanup_interval_secs == 3600


def test_valid_numeric_server_env_values_are_respected(monkeypatch):
    """Valid numeric env values should still override defaults."""
    monkeypatch.setenv("SKTIME_MCP_JOB_MAX_AGE_HOURS", "48")
    monkeypatch.setenv("SKTIME_MCP_JOB_CLEANUP_INTERVAL", "120")

    assert settings.job_max_age_hours == 48
    assert settings.job_cleanup_interval_secs == 120
