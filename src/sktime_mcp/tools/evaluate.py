"""
evaluate tool for sktime MCP.

Executes cross-validation on an estimator.
"""

import logging
from typing import Any

import numpy as np
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


def diagnose_residuals_tool(
    predictions: dict[str, Any] | list[float],
    actuals: dict[str, Any] | list[float],
) -> dict[str, Any]:
    """
    Diagnose residuals by comparing predictions and actuals.

    Args:
        predictions: Forecasted values.
        actuals: Actual observed values.

    Returns:
        Dictionary with statistical metrics (MAE, RMSE, Bias).
    """
    try:
        def extract_values(data):
            if isinstance(data, dict):
                # Handle nested dicts or flat dicts
                try:
                    return np.array(list(data.values()), dtype=float)
                except ValueError:
                    return np.array([float(v) for v in data.values() if isinstance(v, (int, float))])
            return np.array(data, dtype=float)

        y_pred = extract_values(predictions)
        y_true = extract_values(actuals)

        if len(y_pred) != len(y_true):
            return {
                "success": False,
                "error": f"Length mismatch: predictions ({len(y_pred)}) vs actuals ({len(y_true)})",
            }

        residuals = y_true - y_pred
        mae = float(np.mean(np.abs(residuals)))
        mse = float(np.mean(residuals ** 2))
        rmse = float(np.sqrt(mse))
        bias = float(np.mean(residuals))

        return {
            "success": True,
            "metrics": {
                "MAE": mae,
                "RMSE": rmse,
                "Mean_Bias": bias,
            },
            "residuals": [float(r) for r in residuals],
            "diagnosis": (
                f"The model has a mean bias of {bias:.4f}. "
                f"A positive bias means the model under-predicts on average, "
                f"while a negative bias means it over-predicts. "
                f"Average absolute error (MAE) is {mae:.4f}."
            )
        }
    except Exception as e:
        logger.exception("Error during diagnose_residuals")
        return {"success": False, "error": str(e)}
