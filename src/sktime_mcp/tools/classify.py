"""
Classification and regression tools for sktime MCP.

Provides tools for fitting classifiers/regressors and generating predictions.
Supports both built-in demo datasets and custom user-loaded data handles.
"""

import logging
from typing import Any

from sktime_mcp.runtime.executor import get_executor

logger = logging.getLogger(__name__)


def fit_predict_classification_tool(
    estimator_handle: str,
    dataset: str | None = None,
    X_train_handle: str | None = None,
    y_train_handle: str | None = None,
    X_test_handle: str | None = None,
) -> dict[str, Any]:
    """
    Fit a classifier on training data and predict class labels on test data.

    Supports two modes:
    1. Demo dataset mode: provide `dataset` name (e.g., "arrow_head")
    2. Custom data mode: provide `X_train_handle`, `y_train_handle`, `X_test_handle`

    Args:
        estimator_handle: Handle from instantiate_estimator
        dataset: Demo dataset name (arrow_head, gunpoint, basic_motions)
        X_train_handle: Data handle for training features
        y_train_handle: Data handle for training labels
        X_test_handle: Data handle for test features

    Returns:
        Dictionary with predictions and metadata
    """
    executor = get_executor()
    return executor.fit_predict_classification(
        estimator_handle=estimator_handle,
        dataset=dataset,
        X_train_handle=X_train_handle,
        y_train_handle=y_train_handle,
        X_test_handle=X_test_handle,
    )


def fit_predict_regression_tool(
    estimator_handle: str,
    dataset: str | None = None,
    X_train_handle: str | None = None,
    y_train_handle: str | None = None,
    X_test_handle: str | None = None,
) -> dict[str, Any]:
    """
    Fit a regressor on training data and predict target values on test data.

    Supports two modes:
    1. Demo dataset mode: provide `dataset` name (e.g., "covid_3month")
    2. Custom data mode: provide `X_train_handle`, `y_train_handle`, `X_test_handle`

    Args:
        estimator_handle: Handle from instantiate_estimator
        dataset: Demo dataset name (covid_3month)
        X_train_handle: Data handle for training features
        y_train_handle: Data handle for training labels/values
        X_test_handle: Data handle for test features

    Returns:
        Dictionary with predictions and metadata
    """
    executor = get_executor()
    return executor.fit_predict_regression(
        estimator_handle=estimator_handle,
        dataset=dataset,
        X_train_handle=X_train_handle,
        y_train_handle=y_train_handle,
        X_test_handle=X_test_handle,
    )
