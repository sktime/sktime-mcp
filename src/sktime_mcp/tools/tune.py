"""
tune_estimator tool for sktime MCP.

Executes hyperparameter tuning using ForecastingGridSearchCV.
"""

import logging
from typing import Any

from sktime.forecasting.model_selection import (
    ExpandingWindowSplitter,
    ForecastingGridSearchCV,
)

from sktime_mcp.runtime.executor import get_executor
from sktime_mcp.runtime.handles import get_handle_manager

logger = logging.getLogger(__name__)


def tune_estimator_tool(
    estimator_handle: str,
    dataset: str,
    param_grid: dict[str, list[Any]],
    cv_folds: int = 3,
) -> dict[str, Any]:
    """
    Tune an estimator using grid search cross-validation.

    Args:
        estimator_handle: Handle from instantiate_estimator
        dataset: Name of the demo dataset
        param_grid: Dictionary with parameters names (string) as keys
                    and lists of parameter settings to try as values.
        cv_folds: Number of folds for cross-validation

    Returns:
        Dictionary containing:
        - success: bool
        - tuned_handle: A new handle ID for the tuned estimator
        - best_params: The best parameters found
        - best_score: The best CV score achieved
    """
    executor = get_executor()
    handle_manager = get_handle_manager()

    try:
        instance = handle_manager.get_instance(estimator_handle)
        handle_info = handle_manager.get_info(estimator_handle)
        estimator_name = handle_info.estimator_name
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

        # Instantiate and run GridSearch
        gscv = ForecastingGridSearchCV(
            forecaster=instance,
            cv=cv,
            param_grid=param_grid,
        )

        gscv.fit(y=y, X=X)

        best_forecaster = gscv.best_forecaster_
        best_params = gscv.best_params_

        # Save the tuned & fitted estimator as a new handle
        tuned_handle = handle_manager.create_handle(
            estimator_name=f"{estimator_name}_tuned",
            instance=best_forecaster,
            params=best_params,
        )

        # Mark the new handle as fitted since the best_forecaster_ is fitted
        handle_manager.mark_fitted(tuned_handle)

        return {
            "success": True,
            "tuned_handle": tuned_handle,
            "best_params": best_params,
            "best_score": float(gscv.best_score_) if gscv.best_score_ is not None else None,
        }

    except Exception as e:
        logger.exception("Error during tuning")
        return {"success": False, "error": str(e)}
