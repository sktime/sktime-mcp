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

        # Derive step_length so the splitter produces ~cv_folds splits up-front
        # rather than generating every possible split with step_length=1 and
        # trimming post-hoc. This gives both compute savings on large series
        # and more representative, spread-out evaluation windows.
        available = n - initial_window
        if cv_folds is not None and cv_folds > 0 and available > 0:
            step_length = max(1, available // cv_folds)
        else:
            step_length = 1

        cv = ExpandingWindowSplitter(
            initial_window=initial_window, step_length=step_length, fh=[1]
        )

        results = evaluate(forecaster=instance, y=y, X=X, cv=cv)
        # Safety cap: integer division can still overshoot on very small series
        # (e.g. n=10, cv_folds=3 -> step_length=1, 5 folds produced).
        if cv_folds is not None and cv_folds > 0:
            results = results.head(cv_folds)

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
