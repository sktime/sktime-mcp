"""
evaluate tool for sktime MCP.

Executes cross-validation on an estimator.
"""

import logging
from typing import Any

from sktime.forecasting.model_evaluation import evaluate

try:
    from sktime.split import ExpandingWindowSplitter
except ImportError:  # pragma: no cover - sktime < 0.29
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
        folds = max(1, min(int(cv_folds), max(1, n - 1)))
        # Exactly `folds` backtest windows: train grows, last fold uses n-1 obs before last point.
        initial_window = max(1, n - folds)
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
            "cv_folds_run": len(metrics),
            "cv_folds_requested": int(cv_folds),
        }
    except Exception as e:
        logger.exception("Error during evaluate")
        return {"success": False, "error": str(e)}
