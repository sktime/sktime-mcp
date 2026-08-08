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


def _validate_evaluate_inputs(executor, estimator_handle: str, y: str) -> dict[str, Any] | None:
    """Reject non-forecasters and non-Series targets before any CV work.

    Returns an error dict to short-circuit on, or None when inputs are valid.
    Cross-validation here uses an ExpandingWindowSplitter and forecasting
    metrics, which only make sense for a forecaster on a single Series.
    """
    import pandas as pd

    try:
        instance = executor._handle_manager.get_instance(estimator_handle)
    except KeyError:
        return {"success": False, "error": executor._handle_manager.describe_missing(estimator_handle)}

    get_tag = getattr(instance, "get_class_tag", None)
    obj_type = get_tag("object_type", "") if callable(get_tag) else ""
    if obj_type != "forecaster":
        return {
            "success": False,
            "error": (
                f"evaluate cross-validates forecasters, but this handle is a "
                f"{obj_type or 'non-forecaster object'}. Use call_method for "
                "non-forecaster estimators."
            ),
        }

    # Resolve y and confirm it is a numeric Series (not Panel/Hierarchical, and
    # not categorical label data from a classification dataset).
    y_res = executor._resolve_source(y)
    if not y_res["success"]:
        return y_res
    _y = y_res["data"]
    try:
        from sktime.datatypes import check_is_scitype

        is_series = check_is_scitype(_y, scitype="Series", return_metadata=[])[0]
    except Exception:
        is_series = True  # let the downstream run surface an unusual case
    if not is_series:
        return {
            "success": False,
            "error": (
                "evaluate expects a univariate/Series target, but the given y is Panel "
                "or Hierarchical scitype. Forecasting CV is not defined for panel data."
            ),
        }

    import numpy as np

    if isinstance(_y, pd.Series):
        numeric = pd.api.types.is_numeric_dtype(_y)
    elif isinstance(_y, pd.DataFrame):
        numeric = all(pd.api.types.is_numeric_dtype(_y[c]) for c in _y.columns)
    elif isinstance(_y, np.ndarray):
        numeric = np.issubdtype(_y.dtype, np.number)
    else:
        numeric = True
    if not numeric:
        return {
            "success": False,
            "error": (
                f"evaluate needs a numeric forecasting target, but y '{y}' is non-numeric "
                "(categorical/label data — this looks like classification data, not a "
                "forecasting series)."
            ),
        }
    return None


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
    if initial_window is None and cv_folds < 1:
        return {
            "success": False,
            "error": f"cv_folds must be a positive integer, got {cv_folds}",
        }
    if initial_window is not None and initial_window < 1:
        return {
            "success": False,
            "error": f"initial_window must be a positive integer, got {initial_window}",
        }

    executor = get_executor()

    # Scitype validation up front, so async calls reject synchronously instead
    # of burning a background job on invalid inputs (#535).
    invalid = _validate_evaluate_inputs(executor, estimator_handle, y)
    if invalid is not None:
        return invalid

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
        return {
            "success": False,
            "error": executor._handle_manager.describe_missing(estimator_handle),
        }

    y_res = executor._resolve_source(y)
    if not y_res["success"]:
        return y_res
    _y = y_res["data"]

    _X = None
    if X:
        x_res = executor._resolve_source(X, prefer="X")
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
