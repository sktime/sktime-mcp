"""
fit_predict tool for sktime MCP.

Executes complete forecasting workflows.
"""

import logging
from typing import Any, Optional

from sktime_mcp.runtime.executor import get_executor

logger = logging.getLogger(__name__)


def fit_predict_tool(
    estimator_handle: str,
    dataset: str = "",
    horizon: int = 12,
    data_handle: Optional[str] = None,
    background: bool = False,
) -> dict[str, Any]:
    """
    Execute a complete fit-predict workflow.

    Args:
        estimator_handle: Handle from instantiate_estimator
        dataset: Name of demo dataset (e.g., "airline", "sunspots")
        horizon: Forecast horizon (default: 12)
        data_handle: Optional handle from load_data_source for custom data
        background: If True, run in background and return a job_id

    Returns:
        Dictionary with results (sync) or job_id (async)

    Example:
        >>> fit_predict_tool("est_abc123", "airline", horizon=12)
        {
            "success": True,
            "predictions": {1: 450.2, 2: 460.5, ...},
            "horizon": 12
        }
    """
    if data_handle is None and (not dataset or not str(dataset).strip()):
        return {
            "success": False,
            "error": "Provide either dataset (demo name) or data_handle from load_data_source.",
        }
    executor = get_executor()

    if not background:
        return executor.fit_predict(
            estimator_handle, dataset, horizon, data_handle=data_handle
        )

    # Handle background execution
    from sktime_mcp.runtime.jobs import get_job_manager

    job_manager = get_job_manager()

    # Get estimator info for job name
    try:
        handle_info = executor._handle_manager.get_info(estimator_handle)
        estimator_name = handle_info.estimator_name
    except Exception as e:
        logger.warning(f"Could not get estimator name: {e}")
        estimator_name = "Unknown"

    data_source_name = data_handle if data_handle else dataset

    # Create job
    job_id = job_manager.create_job(
        job_type="fit_predict",
        estimator_handle=estimator_handle,
        estimator_name=estimator_name,
        dataset_name=data_source_name,
        horizon=horizon,
        total_steps=3,
    )

    # Schedule the async coroutine
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    coro = executor.fit_predict_async(
        estimator_handle, dataset, horizon, job_id=job_id, data_handle=data_handle
    )
    asyncio.run_coroutine_threadsafe(coro, loop)

    return {
        "success": True,
        "job_id": job_id,
        "message": f"Training job started for {estimator_name} on {data_source_name}. Use check_job_status('{job_id}') to monitor progress.",
        "estimator": estimator_name,
        "dataset": data_source_name,
        "horizon": horizon,
    }


def fit_tool(
    estimator_handle: str,
    dataset: str,
) -> dict[str, Any]:
    """
    Fit an estimator on a dataset.

    Args:
        estimator_handle: Handle from instantiate_estimator
        dataset: Name of demo dataset

    Returns:
        Dictionary with success status
    """
    executor = get_executor()
    data_result = executor.load_dataset(dataset)
    if not data_result["success"]:
        return data_result

    return executor.fit(
        estimator_handle,
        y=data_result["data"],
        X=data_result.get("exog"),
    )


def predict_tool(
    estimator_handle: str,
    horizon: int = 12,
) -> dict[str, Any]:
    """
    Generate predictions from a fitted estimator.

    Args:
        estimator_handle: Handle of a fitted estimator
        horizon: Forecast horizon

    Returns:
        Dictionary with predictions
    """
    executor = get_executor()
    fh = list(range(1, horizon + 1))
    return executor.predict(estimator_handle, fh=fh)



def list_datasets_tool() -> dict[str, Any]:
    """
    List available demo datasets.

    Returns:
        Dictionary with list of dataset names
    """
    executor = get_executor()
    return {
        "success": True,
        "datasets": executor.list_datasets(),
    }
