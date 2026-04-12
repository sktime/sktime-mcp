"""
evaluate tool for sktime MCP.

Executes cross-validation on an estimator.
"""

import logging
from typing import Any

from sktime.forecasting.model_evaluation import evaluate
from sktime.forecasting.model_selection import ExpandingWindowSplitter

from sktime_mcp.runtime.executor import get_executor

logger = logging.getLogger(__name__)


def evaluate_estimator_tool(
    estimator_handle: str,
    dataset: str,
    cv_folds: int = 3,
) -> dict[str, Any]:
    """
    Evaluate an estimator using cross-validation.

    Args:
        estimator_handle: Handle from instantiate_estimator
        dataset: Name of demo dataset
        cv_folds: Number of folds for Splitter

    Returns:
        Dictionary with cross-validation results
    """
    executor = get_executor()

    try:
        instance = executor._handle_manager.get_instance(estimator_handle)
    except KeyError:
        return {"success": False, "error": f"Handle not found: {estimator_handle}"}

    data_result = executor.load_dataset(dataset)
    if not data_result["success"]:
        return data_result

    y = data_result["data"]
    X = data_result.get("exog")

    try:
        n = len(y)
        # Handle small datasets gracefully
        initial_window = max(int(n * 0.5), n - cv_folds * 2)
        if initial_window < 1:
            initial_window = 1

        cv = ExpandingWindowSplitter(initial_window=initial_window, step_length=1, fh=[1])

        results = evaluate(forecaster=instance, y=y, X=X, cv=cv)

        # Convert index or objects to strings suitable for JSON output if needed
        # We drop objects that are complex (like estimator instances themselves) from the output
        if "estimator" in results.columns:
            results = results.drop(columns=["estimator"])

        metrics = results.to_dict(orient="records")

        return {"success": True, "results": metrics, "cv_folds_run": len(metrics)}
    except Exception as e:
        logger.exception("Error during evaluate")
        return {"success": False, "error": str(e)}


def evaluate_estimator_async_tool(
    estimator_handle: str,
    dataset: str,
    cv_folds: int = 3,
) -> dict[str, Any]:
    """
    Evaluate an estimator using cross-validation in the background (non-blocking).

    This tool schedules the evaluation as a background job and returns immediately
    with a job_id. Use check_job_status to monitor progress.

    Args:
        estimator_handle: Handle from instantiate_estimator
        dataset: Name of demo dataset
        cv_folds: Number of folds for Splitter

    Returns:
        Dictionary with:
        - success: bool
        - job_id: Job ID for tracking progress
        - message: Information about the job
    """
    import asyncio
    from sktime_mcp.runtime.jobs import get_job_manager

    executor = get_executor()
    job_manager = get_job_manager()

    try:
        handle_info = executor._handle_manager.get_info(estimator_handle)
        estimator_name = handle_info.estimator_name
    except Exception as e:
        logger.warning(f"Could not get estimator name: {e}")
        estimator_name = "Unknown"

    job_id = job_manager.create_job(
        job_type="evaluate",
        estimator_handle=estimator_handle,
        estimator_name=estimator_name,
        dataset_name=dataset,
        total_steps=2,
    )

    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    coro = executor.evaluate_async(estimator_handle, dataset, cv_folds, job_id=job_id)
    asyncio.run_coroutine_threadsafe(coro, loop)

    return {
        "success": True,
        "job_id": job_id,
        "message": f"Evaluation job started for {estimator_name} on {dataset}. Use check_job_status('{job_id}') to monitor progress.",
        "estimator": estimator_name,
        "dataset": dataset,
        "cv_folds": cv_folds,
    }
