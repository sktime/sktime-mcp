"""
Data loading tools for sktime MCP.

Provides tools for loading data from various sources.
"""

import logging
from typing import Any

from sktime_mcp.runtime.executor import get_executor

logger = logging.getLogger(__name__)


def load_data_source_tool(config: dict[str, Any]) -> dict[str, Any]:
    """
    Load data from any source (pandas, SQL, file, etc.).

    Args:
        config: Data source configuration
            {
                "type": "pandas" | "sql" | "file",
                ... (type-specific configuration)
            }

    Returns:
        Dictionary with:
        - success: bool
        - data_handle: str (handle ID for the loaded data)
        - metadata: dict (information about the data)
        - validation: dict (validation results)

    Examples:
        # Pandas DataFrame
        >>> load_data_source_tool({
        ...     "type": "pandas",
        ...     "data": {"date": [...], "value": [...]},
        ...     "time_column": "date",
        ...     "target_column": "value"
        ... })

        # SQL Database
        >>> load_data_source_tool({
        ...     "type": "sql",
        ...     "connection_string": "postgresql://user:pass@host:5432/db",
        ...     "query": "SELECT date, value FROM sales",
        ...     "time_column": "date",
        ...     "target_column": "value"
        ... })

        # CSV File
        >>> load_data_source_tool({
        ...     "type": "file",
        ...     "path": "/path/to/data.csv",
        ...     "time_column": "date",
        ...     "target_column": "value"
        ... })
    """
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


def fit_predict_with_data_tool(
    estimator_handle: str,
    data_handle: str,
    horizon: int = 12,
) -> dict[str, Any]:
    """
    Fit and predict using custom data.

    Args:
        estimator_handle: Handle from instantiate_estimator
        data_handle: Handle from load_data_source
        horizon: Forecast horizon (default: 12)

    Returns:
        Dictionary with predictions

    Example:
        >>> fit_predict_with_data_tool(
        ...     estimator_handle="est_abc123",
        ...     data_handle="data_xyz789",
        ...     horizon=12
        ... )
    """
    executor = get_executor()
    return executor.fit_predict_with_data(
        estimator_handle,
        data_handle,
        horizon,
    )


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


def load_data_source_async_tool(
    config: dict[str, Any],
) -> dict[str, Any]:
    """
    Load data from any source in the background (non-blocking).

    Schedules the data loading as a background job and returns
    immediately with a job_id. Use check_job_status to monitor
    progress and retrieve the data_handle when done.

    Args:
        config: Data source configuration (same as load_data_source)

    Returns:
        Dictionary with:
        - success: bool
        - job_id: Job ID for tracking progress
        - message: Information about the job

    Example:
        >>> load_data_source_async_tool({
        ...     "type": "file",
        ...     "path": "/path/to/large_data.csv",
        ...     "time_column": "date",
        ...     "target_column": "value"
        ... })
        {
            "success": True,
            "job_id": "abc-123-def-456",
            "message": "Data loading job started..."
        }
    """
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
