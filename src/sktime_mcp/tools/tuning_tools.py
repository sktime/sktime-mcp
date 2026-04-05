"""
Hyperparameter tuning tools for sktime-mcp.

Exposes sktime's ForecastingGridSearchCV, ForecastingRandomizedSearchCV,
and ForecastingOptunaSearchCV as MCP tools.
"""

from typing import Any, Optional

from sktime_mcp.registry.interface import get_registry
from sktime_mcp.runtime.executor import get_executor


def tune_forecaster_tool(
    estimator_handle: str,
    data_handle: str,
    param_grid: dict,
    method: str = "grid",
    fh: int = 12,
    window_length: Optional[int] = None,
    n_iter: int = 10,
    scoring: Optional[str] = None,
) -> dict[str, Any]:
    """
    Tune a forecaster's hyperparameters using single-split evaluation search.

    Args:
        estimator_handle: Handle of the instantiated forecaster to tune
        data_handle: Handle of the loaded dataset
        param_grid: Parameter grid to search, e.g. {"strategy": ["mean", "last"]}
        method: Search method — "grid", "random", or "optuna"
        fh: Forecasting horizon for CV evaluation
        window_length: Training window length for the CV splitter (None = full history)
        n_iter: Number of iterations for random search (ignored for grid/optuna)
        scoring: Metric class name to optimise, e.g. "MeanAbsolutePercentageError"
                 (None = default sktime scoring). Use list_metrics to see options.

    Returns:
        Dictionary with best_params, best_score, and new_handle for the best forecaster
    """
    executor = get_executor()
    return executor.tune_forecaster(
        estimator_handle=estimator_handle,
        data_handle=data_handle,
        param_grid=param_grid,
        method=method,
        fh=fh,
        window_length=window_length,
        n_iter=n_iter,
        scoring=scoring,
    )


def get_param_grid_suggestions_tool(estimator_name: str) -> dict[str, Any]:
    """
    Return a suggested parameter grid for a given estimator.

    Inspects the estimator's hyperparameters and produces a grid of sensible
    values to try during tuning. Use this before calling tune_forecaster to
    build a starting param_grid.

    Args:
        estimator_name: Name of the sktime estimator, e.g. "NaiveForecaster"

    Returns:
        Dictionary with the estimator name and a suggested param_grid
    """
    registry = get_registry()
    node = registry.get_estimator_by_name(estimator_name)
    if node is None:
        return {"success": False, "error": f"Unknown estimator: '{estimator_name}'"}

    suggestions = {}
    for param, info in node.hyperparameters.items():
        # hyperparameters are stored as {'default': value, 'required': bool}
        default = info.get("default") if isinstance(info, dict) else info
        if isinstance(default, bool):
            suggestions[param] = [True, False]
        elif isinstance(default, int):
            candidates = [default - 1, default, default + 1] if default > 0 else [default, default + 1]
            suggestions[param] = list(dict.fromkeys(candidates))
        elif isinstance(default, float):
            suggestions[param] = [default * 0.5, default, default * 2.0]
        elif isinstance(default, str):
            # Single value — user will need to fill in alternatives
            suggestions[param] = [default]
        # Skip None / complex defaults (callables, objects)

    return {
        "success": True,
        "estimator": estimator_name,
        "suggested_param_grid": suggestions,
        "note": (
            "This is a starting suggestion based on parameter types and defaults. "
            "Review and adjust values before passing to tune_forecaster."
        ),
    }
