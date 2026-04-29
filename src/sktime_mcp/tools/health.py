"""
health tool for sktime MCP.

Provides server diagnostics: version info, resource counts,
and job summary statistics.
"""

import logging
import platform
import time
from typing import Any

logger = logging.getLogger(__name__)

# Capture server start time at module-load so it survives across calls.
_SERVER_START_TIME = time.monotonic()


def get_server_health_tool() -> dict[str, Any]:
    """Return a snapshot of the sktime-mcp server health and resource usage.

    Returns:
        Dictionary with:
        - success: bool
        - server_version: sktime-mcp package version
        - python_version: Python interpreter version
        - sktime_version: Installed sktime version
        - platform: OS / architecture string
        - uptime_seconds: Approximate seconds since server module was loaded
        - active_estimator_handles: Number of live estimator handles
        - max_estimator_handles: Configured handle limit
        - active_data_handles: Number of live data handles
        - job_summary: Counts per job status (pending/running/completed/failed/cancelled)
        - registered_tools: Total number of tools listed by the server

    Example::

        >>> get_server_health_tool()
        {
            "success": True,
            "server_version": "0.1.0",
            "python_version": "3.12.3",
            "sktime_version": "0.30.1",
            ...
        }
    """
    result: dict[str, Any] = {"success": True}

    # --- Versions ---
    try:
        from importlib.metadata import version as pkg_version

        result["server_version"] = pkg_version("sktime-mcp")
    except Exception:
        result["server_version"] = "unknown"

    result["python_version"] = platform.python_version()

    try:
        import sktime

        result["sktime_version"] = sktime.__version__
    except Exception:
        result["sktime_version"] = "not installed"

    result["platform"] = platform.platform()

    # --- Uptime ---
    result["uptime_seconds"] = round(time.monotonic() - _SERVER_START_TIME, 2)

    # --- Estimator handles ---
    try:
        from sktime_mcp.runtime.handles import get_handle_manager

        hm = get_handle_manager()
        handles = hm.list_handles()
        result["active_estimator_handles"] = len(handles)
        result["max_estimator_handles"] = hm._max_handles
    except Exception as exc:
        logger.warning(f"Could not read handle manager: {exc}")
        result["active_estimator_handles"] = "unavailable"
        result["max_estimator_handles"] = "unavailable"

    # --- Data handles ---
    try:
        from sktime_mcp.runtime.executor import get_executor

        executor = get_executor()
        result["active_data_handles"] = len(executor._data_handles)
    except Exception as exc:
        logger.warning(f"Could not read data handles: {exc}")
        result["active_data_handles"] = "unavailable"

    # --- Job summary ---
    try:
        from sktime_mcp.runtime.jobs import get_job_manager

        jm = get_job_manager()
        all_jobs = jm.list_jobs()
        summary: dict[str, int] = {}
        for job in all_jobs:
            status = job.get("status", "unknown")
            summary[status] = summary.get(status, 0) + 1
        result["job_summary"] = summary
        result["total_jobs"] = len(all_jobs)
    except Exception as exc:
        logger.warning(f"Could not read job manager: {exc}")
        result["job_summary"] = "unavailable"
        result["total_jobs"] = "unavailable"

    # --- Dependency availability ---
    dep_status: dict[str, bool] = {}
    for pkg in ("numpy", "pandas", "scipy", "matplotlib", "mlflow"):
        try:
            __import__(pkg)
            dep_status[pkg] = True
        except ImportError:
            dep_status[pkg] = False
    result["optional_dependencies"] = dep_status

    return result
