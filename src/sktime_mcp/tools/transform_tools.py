"""
Transformer tools for sktime MCP.

Exposes fit_transform and transform as standalone MCP tools,
enabling LLM-driven preprocessing, feature extraction, and
detrending/deseasonalizing workflows without a downstream predictor.
"""

import logging
from typing import Any, Optional

import pandas as pd

logger = logging.getLogger(__name__)


def _series_to_json(data: Any) -> Any:
    """Convert a pandas Series or DataFrame to a JSON-safe structure."""
    if isinstance(data, pd.Series):
        return {str(k): v for k, v in data.items()}
    if isinstance(data, pd.DataFrame):
        data = data.copy()
        data.index = data.index.astype(str)
        return data.to_dict(orient="list")
    if hasattr(data, "tolist"):
        return data.tolist()
    return data


def _load_y(
    estimator_handle: str,
    dataset: Optional[str],
    data_handle: Optional[str],
) -> tuple:
    """
    Return (y, X, error_dict_or_None).

    Loads the target series from either a demo dataset name or a data handle.
    Returns an error dict as the third element if loading fails.
    """
    from sktime_mcp.runtime.executor import get_executor

    executor = get_executor()

    if dataset is not None:
        result = executor.load_dataset(dataset)
        if not result["success"]:
            return None, None, result
        return result["data"], result.get("exog"), None

    if data_handle is not None:
        if data_handle not in executor._data_handles:
            return None, None, {
                "success": False,
                "error": f"Data handle '{data_handle}' not found",
            }
        info = executor._data_handles[data_handle]
        return info["y"], info.get("X"), None

    return None, None, {
        "success": False,
        "error": "Provide either 'dataset' (demo name) or 'data_handle'",
    }


def fit_transform_tool(
    estimator_handle: str,
    dataset: Optional[str] = None,
    data_handle: Optional[str] = None,
) -> dict[str, Any]:
    """
    Fit a transformer on data and return the transformed result in one step.

    Use this when you want to apply a sktime transformer (e.g. Detrender,
    LogTransformer, Differencer) to a time series without a downstream
    forecaster. The estimator handle is marked as fitted after success.

    Provide exactly one data source:
    - dataset:     demo dataset name (airline, sunspots, lynx, …)
    - data_handle: handle from load_data_source

    Args:
        estimator_handle: Handle from instantiate_estimator or instantiate_pipeline
        dataset:          Demo dataset name (optional)
        data_handle:      Handle from load_data_source (optional)

    Returns:
        Dict with:
        - success:          bool
        - transformed:      JSON-safe representation of the transformed series
        - output_type:      "series" or "dataframe"
        - n_timepoints:     number of time steps in the output
        - estimator_handle: the handle (now marked as fitted)
    """
    from sktime_mcp.runtime.handles import get_handle_manager

    y, X, err = _load_y(estimator_handle, dataset, data_handle)
    if err is not None:
        return err

    handle_manager = get_handle_manager()
    if not handle_manager.exists(estimator_handle):
        return {"success": False, "error": f"Handle not found: {estimator_handle}"}

    instance = handle_manager.get_instance(estimator_handle)

    try:
        transformed = instance.fit_transform(y, X=X) if X is not None else instance.fit_transform(y)

        handle_manager.mark_fitted(estimator_handle)

        output_type = "dataframe" if isinstance(transformed, pd.DataFrame) else "series"

        return {
            "success": True,
            "transformed": _series_to_json(transformed),
            "output_type": output_type,
            "n_timepoints": len(transformed),
            "estimator_handle": estimator_handle,
        }
    except Exception as exc:
        logger.exception("fit_transform_tool failed")
        return {"success": False, "error": str(exc)}


def transform_tool(
    estimator_handle: str,
    dataset: Optional[str] = None,
    data_handle: Optional[str] = None,
) -> dict[str, Any]:
    """
    Apply an already-fitted transformer to data.

    Use this after the transformer has been fitted via fit_transform.
    Allows applying the same fitted transform to new data without re-fitting.

    Provide exactly one data source:
    - dataset:     demo dataset name (airline, sunspots, lynx, …)
    - data_handle: handle from load_data_source

    Args:
        estimator_handle: Handle of a fitted transformer
        dataset:          Demo dataset name (optional)
        data_handle:      Handle from load_data_source (optional)

    Returns:
        Dict with:
        - success:      bool
        - transformed:  JSON-safe representation of the transformed series
        - output_type:  "series" or "dataframe"
        - n_timepoints: number of time steps in the output
    """
    from sktime_mcp.runtime.handles import get_handle_manager

    y, X, err = _load_y(estimator_handle, dataset, data_handle)
    if err is not None:
        return err

    handle_manager = get_handle_manager()
    if not handle_manager.exists(estimator_handle):
        return {"success": False, "error": f"Handle not found: {estimator_handle}"}

    if not handle_manager.is_fitted(estimator_handle):
        return {
            "success": False,
            "error": (
                "Transformer is not fitted. "
                "Call fit_transform first before calling transform."
            ),
        }

    instance = handle_manager.get_instance(estimator_handle)

    try:
        transformed = instance.transform(y, X=X) if X is not None else instance.transform(y)

        output_type = "dataframe" if isinstance(transformed, pd.DataFrame) else "series"

        return {
            "success": True,
            "transformed": _series_to_json(transformed),
            "output_type": output_type,
            "n_timepoints": len(transformed),
        }
    except Exception as exc:
        logger.exception("transform_tool failed")
        return {"success": False, "error": str(exc)}
