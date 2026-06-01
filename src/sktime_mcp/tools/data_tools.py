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
    run_async: bool = False,
) -> dict[str, Any]:
    """
    Load data from any source (pandas, SQL, file, etc.).

    Can run synchronously (blocking) or asynchronously in the background.

    Args:
        config: Data source configuration
            {
                "type": "pandas" | "sql" | "file" | "url",
                ... (type-specific configuration)
            }
        run_async: If True, schedules the loading as a background job and
                   returns a job_id immediately. If False (default), blocks
                   until loaded and returns the data_handle directly.

    Returns:
        Dictionary with:
        If run_async is False:
        - success: bool
        - data_handle: str (handle ID for the loaded data)
        - metadata: dict (information about the data)
        - validation: dict (validation results)

        If run_async is True:
        - success: bool
        - job_id: str (Job ID for tracking progress via check_job_status)
        - message: str (Status message)

    Examples:
        # Synchronous Pandas DataFrame Loading
        >>> load_data_source_tool({
        ...     "type": "pandas",
        ...     "data": {"date": [...], "value": [...]},
        ...     "time_column": "date",
        ...     "target_column": "value"
        ... })

        # Asynchronous CSV File Loading
        >>> load_data_source_tool({
        ...     "type": "file",
        ...     "path": "/path/to/data.csv",
        ...     "time_column": "date",
        ...     "target_column": "value"
        ... }, run_async=True)
    """
    if run_async:
        import asyncio

        from sktime_mcp.runtime.jobs import get_job_manager

        executor = get_executor()
        job_manager = get_job_manager()

        source_type = config.get("type", "unknown")

        # create a background job for data loading
        job_id = job_manager.create_job(
            job_type="data_loading",
            estimator_handle="",
            dataset_name=source_type,
            total_steps=3,  # load, validate, format
        )

        coro = executor.load_data_source_async(config, job_id)

        # Schedule the async coroutine on the event loop
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(coro)
        except RuntimeError:
            # No running event loop (e.g. sync test or CLI environment)
            loop = asyncio.new_event_loop()
            try:
                loop.run_until_complete(coro)
            finally:
                loop.close()

        return {
            "success": True,
            "job_id": job_id,
            "message": (
                f"Data loading job started for source type '{source_type}'. "
                f"Use check_job_status('{job_id}') to monitor progress."
            ),
            "source_type": source_type,
        }
    else:
        executor = get_executor()
        return executor.load_data_source(config)


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
