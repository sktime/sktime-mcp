"""
fit_predict tool for sktime MCP.

Executes complete forecasting workflows.
"""

import logging
from typing import Any

from sktime_mcp.config import settings
from sktime_mcp.runtime.executor import get_executor

logger = logging.getLogger(__name__)


def horizon_cap_error(requested: int) -> str | None:
    """Error text when *requested* steps exceed ``SKTIME_MCP_MAX_HORIZON``, else None.

    predict(horizon=100000) used to forecast every step before truncating the
    response to 500 rows (F-24); the cap rejects such requests before any work.
    """
    cap = settings.max_horizon
    if requested <= cap:
        return None
    return (
        f"horizon={requested} exceeds the server's maximum forecast horizon of "
        f"{cap} steps (SKTIME_MCP_MAX_HORIZON). Request at most {cap} steps, or "
        "raise SKTIME_MCP_MAX_HORIZON in the server environment."
    )


def _validate_horizon(horizon: Any) -> dict[str, Any]:
    """
    Validate the horizon parameter.
    Checks if the horizon parameter is strictly integer or not
    Checks if the horizon parameter is greater than 0 or not
    Checks that it does not exceed ``SKTIME_MCP_MAX_HORIZON`` (F-24)
    """
    warnings = []
    if not isinstance(horizon, int):
        return {
            "valid": False,
            "error": (
                f"'horizon' must be an integer, got {type(horizon).__name__}. "
                f'Example: {{"horizon": 12}}'
            ),
            "warnings": warnings,
        }
    if horizon <= 0:
        return {
            "valid": False,
            "error": f"Invalid horizon={horizon}. horizon must be a positive integer greater than 0.",
            "warnings": warnings,
        }
    cap_error = horizon_cap_error(horizon)
    if cap_error:
        return {"valid": False, "error": cap_error, "warnings": warnings}
    return {"valid": True, "warnings": warnings}


def fit_tool(
    estimator_handle: str,
    X_dataset: str | None = None,
    y_dataset: str | None = None,
    X_handle: str | None = None,
    y_handle: str | None = None,
    fh: Any | None = None,
    run_async: bool = False,
) -> dict[str, Any]:
    """
    Fit an estimator on data.

    X_handle resolves to a data handle's exogenous columns; when only
    y_handle is given and that handle carries X, it is used automatically.
    """
    executor = get_executor()

    # Resolve up front so bad inputs fail synchronously even with run_async.
    resolved = executor._resolve_xy_inputs(
        X_dataset=X_dataset, y_dataset=y_dataset, X_handle=X_handle, y_handle=y_handle
    )
    if not resolved["success"]:
        return resolved
    X, y = resolved["X"], resolved["y"]

    if run_async:
        import asyncio

        from sktime_mcp.runtime.jobs import get_job_manager

        job_manager = get_job_manager()
        try:
            handle_info = executor._handle_manager.get_info(estimator_handle)
            estimator_name = handle_info.estimator_name
        except Exception:
            estimator_name = "Unknown"

        source_name = y_dataset if y_dataset else (y_handle if y_handle else "data")
        job_id = job_manager.create_job(
            job_type="fit",
            estimator_handle=estimator_handle,
            estimator_name=estimator_name,
            dataset_name=source_name,
            total_steps=2,
        )
        task = asyncio.create_task(
            executor.fit_async(
                handle_id=estimator_handle,
                X_dataset=X_dataset,
                y_dataset=y_dataset,
                X_handle=X_handle,
                y_handle=y_handle,
                fh=fh,
                job_id=job_id,
            )
        )
        job_manager.register_task(job_id, task)
        return {"success": True, "job_id": job_id, "status": "running"}
    fit_result = executor.fit(estimator_handle, y=y, X=X, fh=fh)

    if fit_result.get("success") and y_dataset:
        try:
            handle_info = executor._handle_manager.get_info(estimator_handle)
            handle_info.metadata["training_dataset"] = y_dataset
        except Exception as e:
            logger.warning(f"Could not record training dataset: {e}")

    if fit_result.get("success") and resolved["exogenous"]:
        fit_result["exogenous"] = resolved["exogenous"]

    return fit_result


def predict_tool(
    estimator_handle: str,
    horizon: int = 12,
    mode: str = "predict",
    coverage: float | list[float] = 0.9,
    alpha: float | list[float] | None = None,
    X_dataset: str | None = None,
    y_dataset: str | None = None,
    X_handle: str | None = None,
    y_handle: str | None = None,
    run_async: bool = False,
) -> dict[str, Any]:
    """
    Generate predictions from a fitted estimator.

    The result is also registered as a data handle and returned as
    ``prediction_handle`` so it can be plotted (plot_series), written to a
    file (save_data) or scored against a test split (call_method on a metric
    with ``y_true_data_handle``/``y_pred_data_handle``).
    A forecaster fitted with X needs *future* X here: pass X_handle (the
    handle's exogenous columns) or X_dataset explicitly.
    Set run_async=True to run as a background job and return a job_id.
    """
    validation = _validate_horizon(horizon)
    if not validation["valid"]:
        return {
            "success": False,
            "error": validation["error"],
        }

    executor = get_executor()

    # Resolve up front so bad inputs fail synchronously even with run_async.
    resolved = executor._resolve_xy_inputs(
        X_dataset=X_dataset,
        y_dataset=y_dataset,
        X_handle=X_handle,
        y_handle=y_handle,
        auto_X=False,
    )
    if not resolved["success"]:
        return resolved
    X, y = resolved["X"], resolved["y"]

    if run_async:
        import asyncio

        from sktime_mcp.runtime.jobs import get_job_manager

        job_manager = get_job_manager()
        try:
            estimator_name = executor._handle_manager.get_info(estimator_handle).estimator_name
        except Exception:
            estimator_name = "Unknown"

        source_name = y_dataset or y_handle or "data"
        job_id = job_manager.create_job(
            job_type="predict",
            estimator_handle=estimator_handle,
            estimator_name=estimator_name,
            dataset_name=source_name,
            horizon=horizon,
            total_steps=2,
        )
        task = asyncio.create_task(
            executor.predict_async(
                handle_id=estimator_handle,
                horizon=horizon,
                mode=mode,
                coverage=coverage,
                alpha=alpha,
                X_dataset=X_dataset,
                y_dataset=y_dataset,
                X_handle=X_handle,
                y_handle=y_handle,
                job_id=job_id,
            )
        )
        job_manager.register_task(job_id, task)
        return {"success": True, "job_id": job_id, "status": "running"}

    fh = list(range(1, horizon + 1))

    # We must patch executor.predict to accept y as well, to support annotators
    return executor.predict(
        estimator_handle,
        fh=fh,
        X=X,
        y=y,
        mode=mode,
        coverage=coverage,
        alpha=alpha,
    )


def list_datasets_tool() -> dict[str, Any]:
    """
    List available demo datasets.
    """
    executor = get_executor()
    return {
        "success": True,
        "datasets": executor.list_datasets(),
    }


def update_tool(
    estimator_handle: str,
    X_dataset: str | None = None,
    y_dataset: str | None = None,
    X_handle: str | None = None,
    y_handle: str | None = None,
) -> dict[str, Any]:
    executor = get_executor()

    resolved = executor._resolve_xy_inputs(
        X_dataset=X_dataset, y_dataset=y_dataset, X_handle=X_handle, y_handle=y_handle
    )
    if not resolved["success"]:
        return resolved
    X, y = resolved["X"], resolved["y"]

    result = executor.update(estimator_handle, y=y, X=X)
    if result.get("success") and resolved["exogenous"]:
        result["exogenous"] = resolved["exogenous"]
    return result


def get_fitted_params_tool(estimator_handle: str) -> dict[str, Any]:
    executor = get_executor()
    return executor.get_fitted_params(estimator_handle)
