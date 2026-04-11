"""
Forecasting metrics tools for sktime MCP.

Exposes sktime's forecasting performance metrics so LLM agents can
discover available metrics and compute them on explicit y_true / y_pred pairs.
"""

from typing import Any, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Registry of all sktime forecasting metrics
# ---------------------------------------------------------------------------

_METRICS: dict[str, dict[str, Any]] = {
    "MAE": {
        "description": "Mean Absolute Error — average of absolute differences.",
        "lower_is_better": True,
        "scale_dependent": True,
        "import": ("sktime.performance_metrics.forecasting", "MeanAbsoluteError"),
    },
    "MSE": {
        "description": "Mean Squared Error — average of squared differences.",
        "lower_is_better": True,
        "scale_dependent": True,
        "import": ("sktime.performance_metrics.forecasting", "MeanSquaredError"),
        "kwargs": {"square_root": False},
    },
    "RMSE": {
        "description": "Root Mean Squared Error — square root of MSE.",
        "lower_is_better": True,
        "scale_dependent": True,
        "import": ("sktime.performance_metrics.forecasting", "MeanSquaredError"),
        "kwargs": {"square_root": True},
    },
    "MAPE": {
        "description": "Mean Absolute Percentage Error — percentage-based, undefined when y_true=0.",
        "lower_is_better": True,
        "scale_dependent": False,
        "import": (
            "sktime.performance_metrics.forecasting",
            "MeanAbsolutePercentageError",
        ),
        "kwargs": {"symmetric": False},
    },
    "SMAPE": {
        "description": "Symmetric Mean Absolute Percentage Error — symmetric version of MAPE (0–200%).",
        "lower_is_better": True,
        "scale_dependent": False,
        "import": (
            "sktime.performance_metrics.forecasting",
            "MeanAbsolutePercentageError",
        ),
        "kwargs": {"symmetric": True},
    },
    "MASE": {
        "description": (
            "Mean Absolute Scaled Error — MAE scaled by in-sample naive forecast error. "
            "Requires y_train."
        ),
        "lower_is_better": True,
        "scale_dependent": False,
        "requires_y_train": True,
        "import": ("sktime.performance_metrics.forecasting", "MeanAbsoluteScaledError"),
    },
    "RMSSE": {
        "description": (
            "Root Mean Squared Scaled Error — RMSE scaled by in-sample naive. " "Requires y_train."
        ),
        "lower_is_better": True,
        "scale_dependent": False,
        "requires_y_train": True,
        "import": (
            "sktime.performance_metrics.forecasting",
            "MeanSquaredScaledError",
        ),
        "kwargs": {"square_root": True},
    },
    "MedAE": {
        "description": "Median Absolute Error — robust to outliers.",
        "lower_is_better": True,
        "scale_dependent": True,
        "import": ("sktime.performance_metrics.forecasting", "MedianAbsoluteError"),
    },
    "MedSE": {
        "description": "Median Squared Error.",
        "lower_is_better": True,
        "scale_dependent": True,
        "import": ("sktime.performance_metrics.forecasting", "MedianSquaredError"),
        "kwargs": {"square_root": False},
    },
    "MedRMSE": {
        "description": "Median Root Mean Squared Error.",
        "lower_is_better": True,
        "scale_dependent": True,
        "import": ("sktime.performance_metrics.forecasting", "MedianSquaredError"),
        "kwargs": {"square_root": True},
    },
    "MedAPE": {
        "description": "Median Absolute Percentage Error — robust percentage metric.",
        "lower_is_better": True,
        "scale_dependent": False,
        "import": (
            "sktime.performance_metrics.forecasting",
            "MedianAbsolutePercentageError",
        ),
        "kwargs": {"symmetric": False},
    },
    "MedSAPE": {
        "description": "Symmetric Median Absolute Percentage Error.",
        "lower_is_better": True,
        "scale_dependent": False,
        "import": (
            "sktime.performance_metrics.forecasting",
            "MedianAbsolutePercentageError",
        ),
        "kwargs": {"symmetric": True},
    },
    "GMAE": {
        "description": "Geometric Mean Absolute Error.",
        "lower_is_better": True,
        "scale_dependent": True,
        "import": (
            "sktime.performance_metrics.forecasting",
            "GeometricMeanAbsoluteError",
        ),
    },
    "GMSE": {
        "description": "Geometric Mean Squared Error.",
        "lower_is_better": True,
        "scale_dependent": True,
        "import": (
            "sktime.performance_metrics.forecasting",
            "GeometricMeanSquaredError",
        ),
        "kwargs": {"square_root": False},
    },
    "GRMSE": {
        "description": "Geometric Root Mean Squared Error.",
        "lower_is_better": True,
        "scale_dependent": True,
        "import": (
            "sktime.performance_metrics.forecasting",
            "GeometricMeanSquaredError",
        ),
        "kwargs": {"square_root": True},
    },
}


def list_metrics_tool() -> dict[str, Any]:
    """
    List all available sktime forecasting performance metrics.

    Returns a catalogue of metric names, their descriptions, and key
    properties (scale-dependence, whether lower values are better, and
    whether they require training data for normalisation).  The names
    returned here are the exact strings accepted by ``compute_metric``.

    Returns:
        Dictionary with:
        - success: bool
        - metrics: list of metric descriptors (name, description, …)
        - count: total number of available metrics
        - usage_hint: short usage note

    Example:
        >>> list_metrics_tool()
        {
            "success": True,
            "count": 15,
            "metrics": [
                {
                    "name": "MAE",
                    "description": "Mean Absolute Error ...",
                    "lower_is_better": True,
                    "scale_dependent": True,
                    "requires_y_train": False,
                },
                ...
            ],
            "usage_hint": "Pass the metric name to compute_metric to evaluate forecasts."
        }
    """
    metrics = []
    for name, meta in _METRICS.items():
        metrics.append(
            {
                "name": name,
                "description": meta["description"],
                "lower_is_better": meta.get("lower_is_better", True),
                "scale_dependent": meta.get("scale_dependent", True),
                "requires_y_train": meta.get("requires_y_train", False),
            }
        )
    return {
        "success": True,
        "count": len(metrics),
        "metrics": metrics,
        "usage_hint": (
            "Pass the metric 'name' to compute_metric together with y_true and y_pred "
            "to evaluate your forecasts."
        ),
    }


def compute_metric_tool(
    metric: str,
    y_true: list[float],
    y_pred: list[float],
    y_train: Optional[list[float]] = None,
) -> dict[str, Any]:
    """
    Compute a forecasting performance metric on explicit y_true / y_pred values.

    Args:
        metric: Metric name from list_metrics (e.g., "MAE", "RMSE", "MASE").
        y_true: Ground-truth values (list of floats).
        y_pred: Predicted values (list of floats, same length as y_true).
        y_train: In-sample training values required by scale-normalised metrics
                 such as MASE and RMSSE.

    Returns:
        Dictionary with:
        - success: bool
        - metric: name of the metric evaluated
        - value: the numeric score
        - lower_is_better: whether a lower value indicates a better forecast

    Example:
        >>> compute_metric_tool("MAE", y_true=[100, 110], y_pred=[105, 108])
        {"success": True, "metric": "MAE", "value": 3.5, "lower_is_better": True}
    """
    metric_upper = metric.upper()
    if metric_upper not in _METRICS:
        available = sorted(_METRICS.keys())
        return {
            "success": False,
            "error": f"Unknown metric '{metric}'. Use list_metrics to see available options.",
            "available_metrics": available,
        }

    if len(y_true) != len(y_pred):
        return {
            "success": False,
            "error": (
                f"y_true and y_pred must have the same length, "
                f"got {len(y_true)} and {len(y_pred)}."
            ),
        }

    meta = _METRICS[metric_upper]
    requires_y_train = meta.get("requires_y_train", False)
    if requires_y_train and y_train is None:
        return {
            "success": False,
            "error": (
                f"Metric '{metric_upper}' requires y_train (in-sample training values) "
                f"for normalisation. Please supply the y_train argument."
            ),
        }

    try:
        import importlib

        module_path, class_name = meta["import"]
        module = importlib.import_module(module_path)
        MetricClass = getattr(module, class_name)

        kwargs = meta.get("kwargs", {})
        scorer = MetricClass(**kwargs)

        import pandas as pd

        y_true_s = pd.Series(y_true, dtype=float)
        y_pred_s = pd.Series(y_pred, dtype=float)

        if requires_y_train:
            y_train_s = pd.Series(y_train, dtype=float)
            score = scorer(y_true_s, y_pred_s, y_train=y_train_s)
        else:
            score = scorer(y_true_s, y_pred_s)

        # Convert numpy scalar → Python float
        if isinstance(score, (np.floating, np.integer)):
            score = float(score)

        return {
            "success": True,
            "metric": metric_upper,
            "value": score,
            "lower_is_better": meta.get("lower_is_better", True),
        }

    except Exception as e:
        return {
            "success": False,
            "metric": metric_upper,
            "error": str(e),
        }
