"""
evaluate tool for sktime MCP.

Executes cross-validation on an estimator.
"""

import logging
from typing import Any, Optional, Union

from sktime.forecasting.model_evaluation import evaluate
from sktime.forecasting.model_selection import ExpandingWindowSplitter
from sktime.performance_metrics.forecasting import (
    MeanAbsoluteError,
    MeanAbsolutePercentageError,
    MeanSquaredError,
    MedianAbsoluteError,
    GeometricMeanAbsolutePercentageError,
)

from sktime_mcp.runtime.executor import get_executor

logger = logging.getLogger(__name__)


# Mapping of metric names to sktime metric classes
METRIC_NAME_TO_CLASS = {
    "MAE": MeanAbsoluteError,
    "MSE": MeanSquaredError,
    "RMSE": lambda: MeanSquaredError(square_root=True),
    "MAPE": MeanAbsolutePercentageError,
    "SMAPE": GeometricMeanAbsolutePercentageError,
    "MASE": None,  # Requires seasonality parameter, handled specially
    "MedAE": MedianAbsoluteError,
}


def _get_metric_instance(metric_name: str) -> Any:
    """Get a sktime metric instance from a metric name.

    Args:
        metric_name: Name of the metric (MAE, MAPE, MSE, RMSE, SMAPE, MASE, etc.)

    Returns:
        An instance of the appropriate sktime metric class

    Raises:
        ValueError: If the metric is not supported
    """
    metric_name = metric_name.upper()

    if metric_name == "MASE":
        # MASE requires a seasonality parameter - use default sp=1
        from sktime.performance_metrics.forecasting import MeanAbsoluteScaledError
        return MeanAbsoluteScaledError(sp=1)

    if metric_name not in METRIC_NAME_TO_CLASS:
        available = list(METRIC_NAME_TO_CLASS.keys()) + ["MASE"]
        raise ValueError(
            f"Unknown metric: {metric_name}. Available metrics: {available}"
        )

    metric_class_or_func = METRIC_NAME_TO_CLASS[metric_name]

    if callable(metric_class_or_func):
        return metric_class_or_func()
    return metric_class_or_func()


def _extract_metric_value(results: Any, metric_name: str) -> Optional[float]:
    """Extract the metric value from evaluate results.

    The column name format is 'test_<MetricName>' (e.g., 'test_MeanAbsolutePercentageError').

    Args:
        results: DataFrame from sktime evaluate()
        metric_name: Name of the metric to extract

    Returns:
        The metric value as a float, or None if not found
    """
    # Convert column names to find the matching metric
    metric_col = None
    for col in results.columns:
        if col == metric_name or col == f"test_{metric_name}":
            metric_col = col
            break
        # Check case-insensitive match
        col_lower = col.lower()
        metric_lower = metric_name.lower()
        if metric_lower in col_lower or col_lower == f"test_{metric_lower}":
            metric_col = col
            break

    if metric_col is None:
        return None

    # Get the mean of the metric across folds
    values = results[metric_col].dropna()
    if len(values) == 0:
        return None

    return float(values.mean())


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

        return {
            "success": True,
            "results": metrics,
            "cv_folds_run": len(metrics)
        }
    except Exception as e:
        logger.exception("Error during evaluate")
        return {"success": False, "error": str(e)}


def compare_estimators_tool(
    estimator_handles: list[str],
    dataset: Optional[str] = None,
    data_handle: Optional[str] = None,
    metric: str = "MAPE",
    cv_folds: int = 3,
    horizon: int = 12,
) -> dict[str, Any]:
    """
    Compare multiple estimators on the same dataset using cross-validation.

    Runs cross-validation on each estimator with the same dataset and metric,
    then returns results ranked by performance.

    Args:
        estimator_handles: List of estimator handles from instantiate_estimator
        dataset: Name of demo dataset (e.g., "airline", "sunspots")
        data_handle: Alternative to dataset - handle from load_data_source for custom data
        metric: Metric to use for comparison (MAE, MAPE, MSE, RMSE, SMAPE, MASE, MedAE)
        cv_folds: Number of cross-validation folds
        horizon: Forecast horizon for evaluation

    Returns:
        Dictionary with:
        - success: bool
        - ranked: List of dicts with rank, handle, estimator name, score
        - best_handle: Handle of the best estimator
        - best_estimator: Name of the best estimator
        - metric: The metric used for comparison
        - cv_folds: Number of folds run
        - horizon: Forecast horizon used

    Example:
        >>> compare_estimators_tool(["est_abc", "est_def"], dataset="airline", metric="MAPE")
        {
            "success": True,
            "ranked": [
                {"rank": 1, "handle": "est_abc", "estimator": "AutoARIMA", "score": 4.2},
                {"rank": 2, "handle": "est_def", "estimator": "ARIMA", "score": 7.1}
            ],
            "best_handle": "est_abc",
            "best_estimator": "AutoARIMA",
            "metric": "MAPE",
            "cv_folds": 3,
            "horizon": 12
        }
    """
    executor = get_executor()

    # Validate inputs
    if not estimator_handles:
        return {"success": False, "error": "At least one estimator handle is required"}

    if not dataset and not data_handle:
        return {
            "success": False,
            "error": "Either dataset or data_handle must be provided"
        }

    # Load data
    if data_handle:
        if data_handle not in executor._data_handles:
            return {
                "success": False,
                "error": f"Unknown data handle: {data_handle}",
                "available_handles": list(executor._data_handles.keys()),
            }
        data_info = executor._data_handles[data_handle]
        y = data_info["y"]
        X = data_info.get("X")
    else:
        data_result = executor.load_dataset(dataset)
        if not data_result["success"]:
            return data_result
        y = data_result["data"]
        X = data_result.get("exog")

    # Create CV strategy
    try:
        n = len(y)
        initial_window = max(int(n * 0.5), n - cv_folds * 2)
        if initial_window < 1:
            initial_window = 1
        cv = ExpandingWindowSplitter(
            initial_window=initial_window,
            step_length=1,
            fh=list(range(1, horizon + 1)),
        )
    except Exception as e:
        return {"success": False, "error": f"Error creating CV splitter: {str(e)}"}

    # Get metric instance
    try:
        scoring = _get_metric_instance(metric)
    except ValueError as e:
        return {"success": False, "error": str(e)}

    # Evaluate each estimator
    results = []
    errors = []

    for handle in estimator_handles:
        try:
            # Get estimator instance
            try:
                instance = executor._handle_manager.get_instance(handle)
            except KeyError:
                errors.append({"handle": handle, "error": f"Handle not found: {handle}"})
                continue

            # Get estimator name for display
            try:
                handle_info = executor._handle_manager.get_info(handle)
                estimator_name = handle_info.estimator_name
            except Exception:
                estimator_name = "Unknown"

            # Run evaluation
            eval_results = evaluate(
                forecaster=instance,
                y=y,
                X=X,
                cv=cv,
                scoring=scoring,
            )

            # Extract metric value
            score = _extract_metric_value(eval_results, metric)
            if score is None:
                errors.append({
                    "handle": handle,
                    "estimator": estimator_name,
                    "error": f"Could not extract {metric} from results. Available columns: {list(eval_results.columns)}"
                })
                continue

            results.append({
                "handle": handle,
                "estimator": estimator_name,
                "score": score,
            })

        except Exception as e:
            errors.append({"handle": handle, "error": str(e)})

    if not results:
        return {
            "success": False,
            "error": "No estimators could be evaluated successfully",
            "errors": errors,
        }

    # Sort by score (ascending - lower is better for most metrics)
    results.sort(key=lambda x: x["score"])

    # Add ranks
    ranked = []
    for i, result in enumerate(results, 1):
        ranked.append({
            "rank": i,
            "handle": result["handle"],
            "estimator": result["estimator"],
            "score": round(result["score"], 4),
        })

    return {
        "success": True,
        "ranked": ranked,
        "best_handle": ranked[0]["handle"],
        "best_estimator": ranked[0]["estimator"],
        "metric": metric.upper(),
        "cv_folds": cv_folds,
        "horizon": horizon,
        "errors": errors if errors else None,
    }
