"""
evaluate tool for sktime MCP.

Cross-validates an estimator on a dataset.
"""

import asyncio
import logging
from typing import Any

from sktime_mcp.runtime.executor import _resolve_metric_scoring, _run_evaluate, get_executor
from sktime_mcp.runtime.jobs import get_job_manager

logger = logging.getLogger(__name__)


def evaluate_tool(
    estimator_handle: str,
    y: str,
    X: str | None = None,
    cv_folds: int = 3,
    metric: str | None = None,
    initial_window: int | None = None,
    run_async: bool = False,
) -> dict[str, Any]:
    """
    Cross-validate an estimator on a dataset.

    y and X accept data_handle ids or built-in demo dataset names.
    Set run_async=True to run as a background job.
    """
    executor = get_executor()

    if run_async:
        job_manager = get_job_manager()
        try:
            estimator_name = executor._handle_manager.get_info(estimator_handle).estimator_name
        except Exception:
            estimator_name = "Unknown"

        job_id = job_manager.create_job(
            job_type="evaluate",
            estimator_handle=estimator_handle,
            estimator_name=estimator_name,
            dataset_name=y,
            total_steps=3,
        )
        task = asyncio.create_task(
            executor.evaluate_async(
                handle_id=estimator_handle,
                y=y,
                X=X,
                cv_folds=cv_folds,
                metric=metric,
                initial_window=initial_window,
                job_id=job_id,
            )
        )
        job_manager.register_task(job_id, task)
        return {"success": True, "job_id": job_id, "status": "running"}

    try:
        instance = executor._handle_manager.get_instance(estimator_handle)
    except KeyError:
        return {"success": False, "error": f"Handle not found: {estimator_handle}"}

    y_res = executor._resolve_source(y)
    if not y_res["success"]:
        return y_res
    _y = y_res["data"]

    _X = None
    if X:
        x_res = executor._resolve_source(X)
        if not x_res["success"]:
            return x_res
        _X = x_res["data"]

    scoring = None
    if metric:
        scoring = _resolve_metric_scoring(metric)
        if scoring is None:
            return {
                "success": False,
                "error": (
                    f"Unknown metric: {metric}. "
                    "Check available metrics with query_registry(task='metric')."
                ),
            }

    try:
        fold_results, metrics, summary = _run_evaluate(
            instance, _y, _X, cv_folds, scoring, initial_window
        )
    except Exception as e:
        logger.exception("Error during evaluate")
        return {"success": False, "error": str(e)}

    return {
        "success": True,
        "metrics": metrics,
        "fold_results": fold_results,
        "summary": summary,
        "cv_folds_run": len(fold_results),
        "cv_folds_requested": cv_folds,
    }
