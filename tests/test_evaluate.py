"""
Tests for evaluate tool.
"""

import pytest
from sktime.forecasting.naive import NaiveForecaster

from sktime_mcp.runtime.executor import get_executor
from sktime_mcp.tools.evaluate import evaluate_tool


def test_evaluate_tool_basic():
    """evaluate_tool returns fold_results and summary for a simple forecaster."""
    executor = get_executor()
    handle = executor._handle_manager.create_handle("NaiveForecaster", NaiveForecaster(), {})

    try:
        result = evaluate_tool(estimator_handle=handle, y="airline", cv_folds=2)
        assert result["success"], f"Evaluate failed: {result.get('error')}"
        assert "fold_results" in result
        assert result["cv_folds_run"] == 2
        assert len(result["fold_results"]) == 2

        metric_columns = [k for k in result["fold_results"][0] if "test_" in k]
        assert len(metric_columns) > 0
    finally:
        executor._handle_manager.release_handle(handle)


def test_evaluate_with_metric():
    """evaluate_tool with explicit metric name."""
    executor = get_executor()
    handle = executor._handle_manager.create_handle("NaiveForecaster", NaiveForecaster(), {})

    try:
        result = evaluate_tool(
            estimator_handle=handle,
            y="airline",
            cv_folds=2,
            metric="MeanAbsolutePercentageError",
        )
        assert result["success"], f"Evaluate with metric failed: {result.get('error')}"
        assert len(result["fold_results"]) == 2
    finally:
        executor._handle_manager.release_handle(handle)


def test_evaluate_invalid_metric():
    """evaluate_tool with unknown metric returns error."""
    executor = get_executor()
    handle = executor._handle_manager.create_handle("NaiveForecaster", NaiveForecaster(), {})

    try:
        result = evaluate_tool(
            estimator_handle=handle,
            y="airline",
            cv_folds=2,
            metric="NonexistentMetric",
        )
        assert not result["success"]
        assert "Unknown metric" in result["error"]
    finally:
        executor._handle_manager.release_handle(handle)


def test_evaluate_with_data_handle():
    """evaluate_tool accepts y as a data_handle id."""
    executor = get_executor()
    data_res = executor.load_dataset("airline")
    assert data_res["success"]
    executor._data_handles["test_dh"] = {"y": data_res["data"]}
    handle = executor._handle_manager.create_handle("NaiveForecaster", NaiveForecaster(), {})

    try:
        result = evaluate_tool(estimator_handle=handle, y="test_dh", cv_folds=2)
        assert result["success"], f"Evaluate with data_handle failed: {result.get('error')}"
        assert len(result["fold_results"]) == 2
    finally:
        executor._handle_manager.release_handle(handle)
        executor._data_handles.pop("test_dh", None)


def test_evaluate_initial_window_overrides_cv_folds():
    """When initial_window is set, cv_folds is ignored and folds equal n - initial_window."""
    executor = get_executor()
    handle = executor._handle_manager.create_handle("NaiveForecaster", NaiveForecaster(), {})

    try:
        # airline has 144 obs; initial_window=140 gives 4 expanding folds regardless of cv_folds
        result = evaluate_tool(
            estimator_handle=handle,
            y="airline",
            cv_folds=2,
            initial_window=140,
        )
        assert result["success"], f"Evaluate failed: {result.get('error')}"
        assert len(result["fold_results"]) == 4
    finally:
        executor._handle_manager.release_handle(handle)


def test_evaluate_handle_not_found():
    """evaluate_tool returns a clear error when the estimator_handle is unknown."""
    result = evaluate_tool(estimator_handle="does_not_exist", y="airline", cv_folds=2)
    assert not result["success"]
    assert "Handle not found" in result["error"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
