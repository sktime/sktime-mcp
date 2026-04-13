"""
evaluate tool for sktime MCP.

Executes cross-validation on an estimator.
"""

import logging
from typing import Any, Optional

from sktime.forecasting.model_evaluation import evaluate
from sktime.forecasting.model_selection import ExpandingWindowSplitter

from sktime_mcp.runtime.executor import get_executor

logger = logging.getLogger(__name__)


def evaluate_estimator_tool(
    estimator_handle: str,
    dataset: str = "",
    cv_folds: int = 3,
    data_handle: Optional[str] = None,
) -> dict[str, Any]:
    """
    Evaluate an estimator using cross-validation.

    Args:
        estimator_handle: Handle from instantiate_estimator
        dataset: Name of demo dataset (e.g. 'airline'). Ignored if data_handle is provided.
        cv_folds: Number of folds for Splitter
        data_handle: Handle from load_data_source for custom data. Takes priority over dataset.

    Returns:
        Dictionary with cross-validation results
    """
    executor = get_executor()

    try:
        instance = executor._handle_manager.get_instance(estimator_handle)
    except KeyError:
        return {"success": False, "error": f"Handle not found: {estimator_handle}"}

    if data_handle is not None:
        if data_handle not in executor._data_handles:
            return {"success": False, "error": f"Unknown data handle: {data_handle}"}
        data_info = executor._data_handles[data_handle]
        y = data_info["y"]
        X = data_info.get("X")
    else:
        if not dataset:
            return {"success": False, "error": "Provide either 'dataset' (demo name) or 'data_handle'"}
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

        return {
            "success": True,
            "results": metrics,
            "cv_folds_run": len(metrics)
        }
    except Exception as e:
        logger.exception("Error during evaluate")
        return {"success": False, "error": str(e)}
