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
    """Load data from any source (pandas, SQL, file, etc.).

    Can run synchronously (blocking) or asynchronously in the background.

    Parameters
    ----------
    config : dict
        Data source configuration dictionary. Must contain:
        - "type" : str
            Source type: "pandas", "sql", "file", or "url".
        - Additional type-specific configuration keys (e.g. "data", "path",
          "time_column", "target_column").
    run_async : bool, default=False
        If True, schedules the loading as a background job and
        returns a job_id immediately. If False, blocks until loaded
        and returns the data_handle directly.

    Returns
    -------
    dict
        Dictionary containing load results and metadata.
        If run_async is False, contains:
        - "success" : bool
            True if the data was loaded successfully.
        - "data_handle" : str
            The unique handle ID for the loaded data.
        - "metadata" : dict
            Rich metadata including row count, columns, and data type information.
        - "validation" : dict
            Results of indexing and format validation checks.

        If run_async is True, contains:
        - "success" : bool
            True if the background job was scheduled successfully.
        - "job_id" : str
            Unique job ID to monitor progress via check_job_status.
        - "message" : str
            A user-friendly status message.
        - "source_type" : str
            The type of the source requested to load.

    Examples
    --------
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
            task = loop.create_task(coro)
            job_manager.register_task(job_id, task)
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
    """List all available data source types.

    Returns
    -------
    dict
        Dictionary containing available data sources:
        - "success" : bool
            True if the list was retrieved successfully.
        - "sources" : list of str
            List of supported source type names.
        - "descriptions" : dict
            A mapping of source type names to their class and descriptions.
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
    """Release a data handle and free memory.

    Parameters
    ----------
    data_handle : str
        Data handle to release.

    Returns
    -------
    dict
        Dictionary containing success status:
        - "success" : bool
            True if the handle was successfully released, False otherwise.
        - "message" : str, optional
            Detailed status or error message.
    """
    executor = get_executor()
    return executor.release_data_handle(data_handle)
