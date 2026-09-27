"""
Configuration module for sktime-mcp.

Centralizes environment variables and provides sensible defaults.
"""

import os


def _csv_list(name: str, default: list[str]) -> list[str]:
    """Parse a comma-separated env var into a list, or return ``default`` when unset/blank."""
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return list(default)
    return [item.strip() for item in raw.split(",") if item.strip()]


class Settings:
    """Server and runtime configuration settings."""

    # -- Runtime & Server Settings --
    @property
    def log_level(self) -> str:
        """
        Logging level.
        Env Var: SKTIME_MCP_LOG_LEVEL
        Default: "WARNING"
        """
        return os.environ.get("SKTIME_MCP_LOG_LEVEL", "WARNING").upper()

    @property
    def log_path(self) -> str | None:
        """
        Optional file path to output logs to in addition to stderr.
        Env Var: SKTIME_MCP_LOG_PATH
        Default: None
        """
        return os.environ.get("SKTIME_MCP_LOG_PATH")

    # -- Data Formatting --
    @property
    def auto_format(self) -> bool:
        """
        Whether to automatically format time series data upon load.
        Env Var: SKTIME_MCP_AUTO_FORMAT
        Default: True
        """
        return os.environ.get("SKTIME_MCP_AUTO_FORMAT", "true").lower() == "true"

    # -- Job Management --
    @property
    def job_max_age_hours(self) -> int:
        """
        Maximum age in hours before a job is cleaned up.
        Env Var: SKTIME_MCP_JOB_MAX_AGE_HOURS
        Default: 24
        """
        return int(os.environ.get("SKTIME_MCP_JOB_MAX_AGE_HOURS", "24"))

    @property
    def job_cleanup_interval_secs(self) -> int:
        """
        Interval in seconds for periodic job cleanup.
        Env Var: SKTIME_MCP_JOB_CLEANUP_INTERVAL
        Default: 3600
        """
        return int(os.environ.get("SKTIME_MCP_JOB_CLEANUP_INTERVAL", "3600"))

    # -- Memory & Response Budgets --
    @property
    def max_data_handles(self) -> int:
        """
        Maximum number of active data handles to retain in memory.
        Env Var: SKTIME_MCP_MAX_DATA_HANDLES
        Default: 50
        """
        return int(os.environ.get("SKTIME_MCP_MAX_DATA_HANDLES", "50"))

    @property
    def max_response_tokens(self) -> int:
        """
        Maximum tokens allowed per tool response.
        Env Var: SKTIME_MCP_MAX_RESPONSE_TOKENS
        Default: 0
        """
        raw = os.environ.get("SKTIME_MCP_MAX_RESPONSE_TOKENS", "0")
        try:
            return int(raw)
        except ValueError:
            return 0

    # -- HTTP/SSE transport (sktime_mcp.app) --
    @property
    def http_token(self) -> str | None:
        """
        Bearer token every HTTP/SSE request must present.
        Env Var: SKTIME_MCP_HTTP_TOKEN
        Default: None (the app refuses to start unless SKTIME_MCP_HTTP_INSECURE=true)
        """
        return os.environ.get("SKTIME_MCP_HTTP_TOKEN", "").strip() or None

    @property
    def http_insecure(self) -> bool:
        """
        Serve HTTP/SSE without a bearer token. Never use outside an isolated network.
        Env Var: SKTIME_MCP_HTTP_INSECURE
        Default: False
        """
        return os.environ.get("SKTIME_MCP_HTTP_INSECURE", "false").lower() == "true"

    @property
    def http_allowed_hosts(self) -> list[str]:
        """
        Comma-separated ``Host`` header values accepted by the HTTP/SSE app
        (DNS-rebinding protection). ``name:*`` allows any port.
        Env Var: SKTIME_MCP_HTTP_ALLOWED_HOSTS
        Default: localhost and 127.0.0.1 on any port
        """
        return _csv_list(
            "SKTIME_MCP_HTTP_ALLOWED_HOSTS",
            ["localhost", "localhost:*", "127.0.0.1", "127.0.0.1:*"],
        )

    @property
    def http_allowed_origins(self) -> list[str]:
        """
        Comma-separated ``Origin`` header values accepted by the HTTP/SSE app
        (DNS-rebinding protection and CORS). ``scheme://name:*`` allows any port.
        Env Var: SKTIME_MCP_HTTP_ALLOWED_ORIGINS
        Default: http://localhost and http://127.0.0.1 on any port
        """
        return _csv_list(
            "SKTIME_MCP_HTTP_ALLOWED_ORIGINS",
            ["http://localhost", "http://localhost:*", "http://127.0.0.1", "http://127.0.0.1:*"],
        )

    @property
    def http_disable_run_command(self) -> bool:
        """
        Hide ``run_command`` from clients connected over HTTP/SSE (stdio is unaffected).
        Env Var: SKTIME_MCP_HTTP_DISABLE_RUN_COMMAND
        Default: True
        """
        return os.environ.get("SKTIME_MCP_HTTP_DISABLE_RUN_COMMAND", "true").lower() == "true"


settings = Settings()
