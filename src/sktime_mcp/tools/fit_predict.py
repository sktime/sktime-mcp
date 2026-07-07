"""
fit_predict tool for sktime MCP.

Executes complete forecasting workflows.
"""

import logging
from typing import Any

from sktime_mcp.runtime.executor import get_executor

logger = logging.getLogger(__name__)


def _validate_horizon(horizon: Any) -> dict[str, Any]:
    """
    Validate the horizon parameter.
    Checks if the horizon parameter is strictly integer or not
    Checks if the horizon parameter is greater than 0 or not
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
    """
    executor = get_executor()

    # We must resolve y and X from the provided handles/datasets
    X = None
    y = None

    if X_handle:
        if X_handle not in executor._data_handles:
            return {"success": False, "error": f"Unknown X data handle: {X_handle}"}
        X = executor._data_handles[X_handle]["y"]  # 'y' stores the primary object

    if y_handle:
        if y_handle not in executor._data_handles:
            return {"success": False, "error": f"Unknown y data handle: {y_handle}"}
        y = executor._data_handles[y_handle]["y"]

    if X_dataset and X_dataset == y_dataset:
        data_res = executor.load_dataset(X_dataset)
        if not data_res["success"]:
            return data_res
        if data_res.get("exog") is not None:
            X = data_res["data"]
            y = data_res["exog"]
        else:
            y = data_res["data"]
            X = None
    else:
        if X_dataset:
            data_res = executor.load_dataset(X_dataset)
            if not data_res["success"]:
                return data_res
            X = data_res["data"]

        if y_dataset:
            data_res = executor.load_dataset(y_dataset)
            if not data_res["success"]:
                return data_res
            y = data_res["data"]

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
        asyncio.create_task(
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
        return {"success": True, "job_id": job_id, "status": "running"}
    fit_result = executor.fit(estimator_handle, y=y, X=X, fh=fh)

    if fit_result.get("success") and y_dataset:
        try:
            handle_info = executor._handle_manager.get_info(estimator_handle)
            handle_info.metadata["training_dataset"] = y_dataset
        except Exception as e:
            logger.warning(f"Could not record training dataset: {e}")

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
) -> dict[str, Any]:
    """
    Generate predictions from a fitted estimator.
    """
    validation = _validate_horizon(horizon)
    if not validation["valid"]:
        return {
            "success": False,
            "error": validation["error"],
        }

    executor = get_executor()
    X = None
    y = None

    if X_handle:
        if X_handle not in executor._data_handles:
            return {"success": False, "error": f"Unknown X data handle: {X_handle}"}
        X = executor._data_handles[X_handle]["y"]

    if y_handle:
        if y_handle not in executor._data_handles:
            return {"success": False, "error": f"Unknown y data handle: {y_handle}"}
        y = executor._data_handles[y_handle]["y"]

    if X_dataset and X_dataset == y_dataset:
        data_res = executor.load_dataset(X_dataset)
        if not data_res["success"]:
            return data_res
        X = data_res["data"]
        y = data_res.get("exog")
    else:
        if X_dataset:
            data_res = executor.load_dataset(X_dataset)
            if not data_res["success"]:
                return data_res
            X = data_res["data"]

        if y_dataset:
            data_res = executor.load_dataset(y_dataset)
            if not data_res["success"]:
                return data_res
            y = data_res["data"]

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

    X = None
    y = None

    if X_handle:
        if X_handle not in executor._data_handles:
            return {"success": False, "error": f"Unknown X data handle: {X_handle}"}
        X = executor._data_handles[X_handle]["y"]

    if y_handle:
        if y_handle not in executor._data_handles:
            return {"success": False, "error": f"Unknown y data handle: {y_handle}"}
        y = executor._data_handles[y_handle]["y"]

    if X_dataset and X_dataset == y_dataset:
        data_res = executor.load_dataset(X_dataset)
        if not data_res["success"]:
            return data_res
        X = data_res["data"]
        y = data_res.get("exog")
    else:
        if X_dataset:
            data_res = executor.load_dataset(X_dataset)
            if not data_res["success"]:
                return data_res
            X = data_res["data"]

        if y_dataset:
            data_res = executor.load_dataset(y_dataset)
            if not data_res["success"]:
                return data_res
            y = data_res["data"]

    return executor.update(estimator_handle, y=y, X=X)


def get_fitted_params_tool(estimator_handle: str) -> dict[str, Any]:
    executor = get_executor()
    return executor.get_fitted_params(estimator_handle)
