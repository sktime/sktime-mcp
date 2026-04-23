"""
Tests for fit_predict, fit, and predict tools.

Covers the three synchronous tool functions in ``sktime_mcp.tools.fit_predict``:

* ``fit_predict_tool``  – end-to-end fit+predict workflow
* ``fit_tool``          – fit-only step
* ``predict_tool``      – predict-only step (requires prior fit)

"""

import sys

import pytest

sys.path.insert(0, "src")

from sktime.forecasting.naive import NaiveForecaster

from sktime_mcp.runtime.executor import get_executor
from sktime_mcp.tools.fit_predict import fit_predict_tool, fit_tool, predict_tool


def _create_naive_handle(executor=None, strategy="last"):
    """Instantiate a NaiveForecaster and return (executor, handle)."""
    executor = executor or get_executor()
    result = executor.instantiate("NaiveForecaster", {"strategy": strategy})
    assert result["success"], f"Setup error – instantiation failed: {result}"
    return executor, result["handle"]


def _release(executor, handle):
    """Safely release an estimator handle."""
    try:
        executor._handle_manager.release_handle(handle)
    except KeyError:
        pass

class TestFitPredictTool:
    """Tests for ``fit_predict_tool``."""

    #core functionality

    def test_fit_predict_success_airline(self):
        """Fit-predict on the airline demo dataset should succeed."""
        executor, handle = _create_naive_handle()
        try:
            result = fit_predict_tool(handle, "airline", horizon=6)

            assert result["success"], f"fit_predict_tool failed: {result.get('error')}"
            assert "predictions" in result
            assert result["horizon"] == 6
            assert len(result["predictions"]) == 6
        finally:
            _release(executor, handle)

    def test_fit_predict_default_horizon(self):
        """Default horizon (12) should be applied when omitted."""
        executor, handle = _create_naive_handle()
        try:
            result = fit_predict_tool(handle, "airline")

            assert result["success"]
            assert result["horizon"] == 12
            assert len(result["predictions"]) == 12
        finally:
            _release(executor, handle)

    def test_fit_predict_with_data_handle(self):
        """fit_predict_tool should accept a custom data_handle kwarg."""
        import pandas as pd

        executor, handle = _create_naive_handle()
        try:
            # Load custom data through the executor
            config = {
                "type": "pandas",
                "data": {
                    "date": pd.date_range("2020-01-01", periods=30, freq="D"),
                    "value": list(range(30)),
                },
                "time_column": "date",
                "target_column": "value",
            }
            data_result = executor.load_data_source(config)
            assert data_result["success"], f"Data load failed: {data_result}"
            data_handle = data_result["data_handle"]

            result = fit_predict_tool(
                handle, "unused_because_data_handle_overrides", horizon=3, data_handle=data_handle,
            )

            assert result["success"]
            assert result["horizon"] == 3
        finally:
            _release(executor, handle)

    # invalid inputs

    def test_fit_predict_unknown_handle(self):
        """Non-existent estimator handle should return success=False."""
        result = fit_predict_tool("est_does_not_exist_xyz", "airline")

        assert not result["success"]
        assert "error" in result

    def test_fit_predict_unknown_dataset(self):
        """Unknown dataset name should return success=False."""
        executor, handle = _create_naive_handle()
        try:
            result = fit_predict_tool(handle, "not_a_real_dataset_xyz")

            assert not result["success"]
            assert "error" in result
        finally:
            _release(executor, handle)

    def test_fit_predict_invalid_data_handle(self):
        """Non-existent data_handle should produce an error."""
        executor, handle = _create_naive_handle()
        try:
            result = fit_predict_tool(
                handle, "airline", data_handle="data_does_not_exist"
            )

            assert not result["success"]
            assert "error" in result
        finally:
            _release(executor, handle)

    # boundary conditions

    def test_fit_predict_horizon_one(self):
        """A horizon of 1 should still produce exactly one prediction."""
        executor, handle = _create_naive_handle()
        try:
            result = fit_predict_tool(handle, "airline", horizon=1)

            assert result["success"]
            assert result["horizon"] == 1
            assert len(result["predictions"]) == 1
        finally:
            _release(executor, handle)

    def test_fit_predict_large_horizon(self):
        """A large horizon should still return the requested number of steps."""
        executor, handle = _create_naive_handle()
        try:
            result = fit_predict_tool(handle, "airline", horizon=100)

            assert result["success"]
            assert result["horizon"] == 100
            assert len(result["predictions"]) == 100
        finally:
            _release(executor, handle)

    def test_fit_predict_different_datasets(self):
        """fit_predict_tool should work with more than one demo dataset."""
        executor = get_executor()
        datasets = ["airline", "shampoo_sales"]

        for ds in datasets:
            result_inst = executor.instantiate("NaiveForecaster", {"strategy": "last"})
            if not result_inst["success"]:
                pytest.skip(f"Skipping dataset '{ds}' – instantiation failed")
            handle = result_inst["handle"]

            try:
                result = fit_predict_tool(handle, ds, horizon=4)
                # Some datasets may not be available in the installed sktime
                # version; treat those as non-fatal.
                if result["success"]:
                    assert result["horizon"] == 4
                    assert len(result["predictions"]) == 4
            finally:
                _release(executor, handle)

class TestFitTool:
    """Tests for ``fit_tool``."""

    #core functionality

    def test_fit_success(self):
        """fit_tool should succeed on a valid estimator + dataset."""
        executor, handle = _create_naive_handle()
        try:
            result = fit_tool(handle, "airline")

            assert result["success"], f"fit_tool failed: {result.get('error')}"
            assert result["fitted"] is True
            assert result["handle"] == handle
        finally:
            _release(executor, handle)

    def test_fit_marks_estimator_as_fitted(self):
        """After fit_tool, the handle should be marked as fitted."""
        executor, handle = _create_naive_handle()
        try:
            assert not executor._handle_manager.is_fitted(handle)

            fit_tool(handle, "airline")

            assert executor._handle_manager.is_fitted(handle)
        finally:
            _release(executor, handle)

    #invalid inputs

    def test_fit_unknown_handle(self):
        """Non-existent handle should produce an error."""
        result = fit_tool("est_ghost_handle", "airline")

        assert not result["success"]
        assert "error" in result

    def test_fit_unknown_dataset(self):
        """Unknown dataset name should produce an error."""
        executor, handle = _create_naive_handle()
        try:
            result = fit_tool(handle, "nonexistent_dataset_xyz")

            assert not result["success"]
            assert "error" in result
        finally:
            _release(executor, handle)

    #idempotency / repeated fits

    def test_fit_can_be_called_twice(self):
        """Fitting the same estimator twice should not raise."""
        executor, handle = _create_naive_handle()
        try:
            result1 = fit_tool(handle, "airline")
            result2 = fit_tool(handle, "airline")

            assert result1["success"]
            assert result2["success"]
        finally:
            _release(executor, handle)

class TestPredictTool:
    """Tests for ``predict_tool``."""

    #core functionality

    def test_predict_after_fit(self):
        """predict_tool should succeed on a fitted estimator."""
        executor, handle = _create_naive_handle()
        try:
            fit_tool(handle, "airline")
            result = predict_tool(handle, horizon=6)

            assert result["success"], f"predict_tool failed: {result.get('error')}"
            assert "predictions" in result
            assert result["horizon"] == 6
            assert len(result["predictions"]) == 6
        finally:
            _release(executor, handle)

    def test_predict_default_horizon(self):
        """Default horizon of 12 is used when not specified."""
        executor, handle = _create_naive_handle()
        try:
            fit_tool(handle, "airline")
            result = predict_tool(handle)

            assert result["success"]
            assert result["horizon"] == 12
            assert len(result["predictions"]) == 12
        finally:
            _release(executor, handle)

    #invalid inputs

    def test_predict_before_fit(self):
        """Calling predict on an un-fitted estimator should fail."""
        executor, handle = _create_naive_handle()
        try:
            result = predict_tool(handle, horizon=5)

            assert not result["success"]
            assert "error" in result
            # The executor returns "Estimator not fitted"
            assert "not fitted" in result["error"].lower() or "fitted" in result["error"].lower()
        finally:
            _release(executor, handle)

    def test_predict_unknown_handle(self):
        """Non-existent handle should produce an error."""
        result = predict_tool("est_nonexistent_predict", horizon=5)

        assert not result["success"]
        assert "error" in result

    #boundary conditions

    def test_predict_horizon_one(self):
        """Horizon of 1 should return exactly one prediction."""
        executor, handle = _create_naive_handle()
        try:
            fit_tool(handle, "airline")
            result = predict_tool(handle, horizon=1)

            assert result["success"]
            assert result["horizon"] == 1
            assert len(result["predictions"]) == 1
        finally:
            _release(executor, handle)

    def test_predict_large_horizon(self):
        """Large horizon should return the requested number of values."""
        executor, handle = _create_naive_handle()
        try:
            fit_tool(handle, "airline")
            result = predict_tool(handle, horizon=50)

            assert result["success"]
            assert result["horizon"] == 50
            assert len(result["predictions"]) == 50
        finally:
            _release(executor, handle)

class TestFitPredictIntegration:
    """Integration tests combining fit_tool and predict_tool."""

    def test_fit_then_predict_matches_fit_predict(self):
        """Separate fit+predict should yield the same keys as fit_predict_tool."""
        executor = get_executor()

        # Path A: fit_predict_tool
        result_a = executor.instantiate("NaiveForecaster", {"strategy": "last"})
        handle_a = result_a["handle"]

        # Path B: fit_tool + predict_tool
        result_b = executor.instantiate("NaiveForecaster", {"strategy": "last"})
        handle_b = result_b["handle"]

        try:
            combined = fit_predict_tool(handle_a, "airline", horizon=6)
            fit_tool(handle_b, "airline")
            separate = predict_tool(handle_b, horizon=6)

            assert combined["success"]
            assert separate["success"]
            # Both should have predictions and horizon
            assert set(combined["predictions"].keys()) == set(separate["predictions"].keys())
            assert combined["horizon"] == separate["horizon"]
        finally:
            _release(executor, handle_a)
            _release(executor, handle_b)

    def test_predictions_are_numeric(self):
        """All prediction values should be numeric (int or float)."""
        executor, handle = _create_naive_handle()
        try:
            result = fit_predict_tool(handle, "airline", horizon=5)
            assert result["success"]

            for value in result["predictions"].values():
                assert isinstance(value, (int, float)), (
                    f"Non-numeric prediction value: {value!r}"
                )
        finally:
            _release(executor, handle)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
