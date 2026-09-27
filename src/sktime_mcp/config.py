"""
Configuration module for sktime-mcp.

Centralizes environment variables and provides sensible defaults.

Every ``Settings`` property reads its environment variable at access time, so
changes to the environment take effect on the next read without a reload.
"""

import logging
import os

logger = logging.getLogger(__name__)


def _get_int_env(name: str, default: int, minimum: int | None = None) -> int:
    """Return the integer value of env var *name*, or *default* when unusable.

    A value that is not an integer, or that is below *minimum* (when given),
    is rejected with a logged warning and *default* is returned instead, so a
    typo in the environment can never raise at the call site.
    """
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default

    try:
        value = int(raw_value)
    except (TypeError, ValueError):
        logger.warning(
            "Invalid %s=%r (not an integer); using default %d.", name, raw_value, default
        )
        return default

    if minimum is not None and value < minimum:
        logger.warning(
            "Invalid %s=%r (must be >= %d); using default %d.", name, raw_value, minimum, default
        )
        return default

    return value


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
        Maximum age in hours before a finished job is cleaned up.
        Env Var: SKTIME_MCP_JOB_MAX_AGE_HOURS
        Default: 24 (minimum: 1)
        """
        return _get_int_env("SKTIME_MCP_JOB_MAX_AGE_HOURS", 24, minimum=1)

    @property
    def job_cleanup_interval_secs(self) -> int:
        """
        Interval in seconds for periodic job cleanup.
        Env Var: SKTIME_MCP_JOB_CLEANUP_INTERVAL
        Default: 3600 (minimum: 1)
        """
        return _get_int_env("SKTIME_MCP_JOB_CLEANUP_INTERVAL", 3600, minimum=1)

    # -- Memory & Response Budgets --
    @property
    def max_data_handles(self) -> int:
        """
        Maximum number of active data handles to retain in memory.
        Env Var: SKTIME_MCP_MAX_DATA_HANDLES
        Default: 50 (minimum: 1)
        """
        return _get_int_env("SKTIME_MCP_MAX_DATA_HANDLES", 50, minimum=1)

    @property
    def max_response_tokens(self) -> int:
        """
        Maximum tokens allowed per tool response; 0 means unlimited.
        Env Var: SKTIME_MCP_MAX_RESPONSE_TOKENS
        Default: 0 (minimum: 0)
        """
        return _get_int_env("SKTIME_MCP_MAX_RESPONSE_TOKENS", 0, minimum=0)


settings = Settings()
