"""
Data loading tools for sktime MCP.

Provides tools for loading data from various sources.
"""

import logging
from typing import Any

from sktime_mcp.runtime.executor import get_executor

logger = logging.getLogger(__name__)


def load_data_source_tool(
    config: dict[str, Any],
    background: bool = False,
) -> dict[str, Any]:
    """
    Load data from any source (Sync or Async).

    Supported source types: pandas, SQL, file, url.

    Args:
        config: Data source configuration
        background: If True, runs the loading in the background and returns a job_id

    Returns:
        If background=False (default):
            Dictionary with data_handle and metadata.
        If background=True:
            Dictionary with success status, job_id, and tracking message.

    Examples:
        # Sync loading (default)
        >>> load_data_source_tool({"type": "file", "path": "data.csv"})

        # Async loading
        >>> load_data_source_tool({"type": "file", "path": "large.csv"}, background=True)
    """
    executor = get_executor()

    if not background:
        return executor.load_data_source(config)

    # --- Async Logic ---
    import asyncio

    from sktime_mcp.runtime.jobs import get_job_manager

    job_manager = get_job_manager()
    source_type = config.get("type", "unknown")

    # create a background job for data loading
    job_id = job_manager.create_job(
        job_type="data_loading",
        estimator_handle="",
        dataset_name=source_type,
        total_steps=3,  # load, validate, format
    )

    # schedule on event loop
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    coro = executor.load_data_source_async(config, job_id)
    asyncio.run_coroutine_threadsafe(coro, loop)

    return {
        "success": True,
        "job_id": job_id,
        "message": (
            f"Data loading job started for source type '{source_type}'. "
            f"Use check_job_status('{job_id}') to monitor progress."
        ),
        "source_type": source_type,
    }


def list_data_sources_tool() -> dict[str, Any]:
    """
    List all available data source types.

    Returns:
        Dictionary with:
        - success: bool
        - sources: list of available source types
        - descriptions: dict with descriptions for each source type
    """
    from sktime_mcp.data import DataSourceRegistry

    sources = DataSourceRegistry.list_adapters()

    # Get descriptions for each source
    descriptions = {}
    for source_type in sources:
        info = DataSourceRegistry.get_adapter_info(source_type)
        descriptions[source_type] = {
            "class": info["class"],
            "description": info["docstring"].split("\n")[0] if info["docstring"] else "",
        }

    return {
        "success": True,
        "sources": sources,
        "descriptions": descriptions,
    }


def release_data_handle_tool(data_handle: str) -> dict[str, Any]:
    """
    Release a data handle and free memory.

    Args:
        data_handle: Data handle to release

    Returns:
        Dictionary with success status
    """
    executor = get_executor()
    return executor.release_data_handle(data_handle)


