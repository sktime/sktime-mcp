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

        - ``"type"`` (str) -- source type: ``"pandas"``, ``"sql"``, ``"file"``,
          or ``"url"``.
        - Additional type-specific configuration keys (e.g. ``"data"``,
          ``"path"``, ``"time_column"``, ``"target_column"``).
    run_async : bool, default=False
        If True, schedules the loading as a background job and
        returns a job_id immediately. If False, blocks until loaded
        and returns the data_handle directly.

    Returns
    -------
    dict
        Dictionary containing load results and metadata.

        If ``run_async`` is False, contains:

        - ``"success"`` (bool) -- True if the data was loaded successfully.
        - ``"data_handle"`` (str) -- the unique handle ID for the loaded data.
        - ``"metadata"`` (dict) -- rich metadata including row count, columns,
          and data type information.
        - ``"validation"`` (dict) -- results of indexing and format validation
          checks.

        If ``run_async`` is True, contains:

        - ``"success"`` (bool) -- True if the background job was scheduled
          successfully.
        - ``"job_id"`` (str) -- unique job ID to monitor progress via
          ``check_job_status``.
        - ``"message"`` (str) -- a user-friendly status message.
        - ``"source_type"`` (str) -- the type of the source requested to load.

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
    executor = get_executor()
    if not run_async:
        return executor.load_data_source(config)

    from sktime_mcp.runtime.jobs import get_job_manager, start_background_job

    source_type = config.get("type", "unknown")
    scheduled = start_background_job(
        get_job_manager(),
        lambda job_id: executor.load_data_source_async(config, job_id),
        job_type="data_loading",
        estimator_handle="",
        dataset_name=source_type,
        total_steps=3,  # load, validate, format
    )
    if not scheduled["success"]:
        return scheduled

    job_id = scheduled["job_id"]
    return {
        **scheduled,
        "message": (
            f"Data loading job started for source type '{source_type}'. "
            f"Use check_job_status('{job_id}') to monitor progress."
        ),
        "source_type": source_type,
    }


def list_data_sources_tool() -> dict[str, Any]:
    """List all available data source types.

    Returns
    -------
    dict
        Dictionary containing available data sources:

        - ``"success"`` (bool) -- True if the list was retrieved successfully.
        - ``"sources"`` (list of str) -- List of supported source type names.
        - ``"descriptions"`` (dict) -- A mapping of source type names to their class and
          descriptions.
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

        - ``"success"`` (bool) -- True if the handle was successfully released, False otherwise.
        - ``"message"`` (str, optional) -- Detailed status or error message.
    """
    executor = get_executor()
    return executor.release_data_handle(data_handle)
