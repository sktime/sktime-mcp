"""
Supervised execution tools for sktime MCP.

Provides dedicated classification and regression workflows.
"""

from typing import Any, Optional

from sktime_mcp.runtime.executor import get_executor


def fit_predict_classification_tool(
    estimator_handle: str,
    train_data_handle: str,
    predict_data_handle: Optional[str] = None,
    return_probabilities: bool = False,
) -> dict[str, Any]:
    """
    Fit a classifier on supervised data and predict class labels.

    Args:
        estimator_handle: Handle from instantiate_estimator or instantiate_pipeline
        train_data_handle: Data handle containing target and feature columns
        predict_data_handle: Optional feature-only or supervised data handle for inference.
            If omitted, predictions are generated on the training features.
        return_probabilities: Whether to include predict_proba output when available

    Returns:
        Dictionary with predicted labels and optional probabilities
    """
    executor = get_executor()
    return executor.fit_predict_classification_with_data(
        estimator_handle=estimator_handle,
        train_data_handle=train_data_handle,
        predict_data_handle=predict_data_handle,
        return_probabilities=return_probabilities,
    )


def fit_predict_regression_tool(
    estimator_handle: str,
    train_data_handle: str,
    predict_data_handle: Optional[str] = None,
) -> dict[str, Any]:
    """
    Fit a regressor on supervised data and predict numeric targets.

    Args:
        estimator_handle: Handle from instantiate_estimator or instantiate_pipeline
        train_data_handle: Data handle containing target and feature columns
        predict_data_handle: Optional feature-only or supervised data handle for inference.
            If omitted, predictions are generated on the training features.

    Returns:
        Dictionary with predicted numeric values
    """
    executor = get_executor()
    return executor.fit_predict_regression_with_data(
        estimator_handle=estimator_handle,
        train_data_handle=train_data_handle,
        predict_data_handle=predict_data_handle,
    )
