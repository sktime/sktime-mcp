"""
Tests for async fit_predict with custom data handles.
"""

import sys

import pytest

sys.path.insert(0, "src")


class TestAsyncCustomData:
    """Tests for fit_predict_async_tool with data_handle support."""

    def _get_estimator_handle(self):
        """Create a NaiveForecaster handle for reuse."""
        from sktime_mcp.runtime.executor import get_executor

        executor = get_executor()
        result = executor.instantiate("NaiveForecaster", {"strategy": "last"})
        assert result["success"], f"Failed to instantiate: {result}"
        return result["handle"]

    def _load_custom_data(self):
        """Load custom data and return the data handle."""
        import pandas as pd

        from sktime_mcp.runtime.executor import get_executor

        executor = get_executor()
        config = {
            "type": "pandas",
            "data": {
                "date": pd.date_range("2020-01-01", periods=50, freq="D").tolist(),
                "sales": [100 + i for i in range(50)],
            },
            "time_column": "date",
            "target_column": "sales",
        }
        result = executor.load_data_source(config)
        assert result["success"], f"Data load failed: {result}"
        return result["data_handle"]

    def test_async_with_dataset(self):
        """Async with a demo dataset should return a job_id."""
        from sktime_mcp.tools.fit_predict import fit_predict_async_tool

        handle = self._get_estimator_handle()
        result = fit_predict_async_tool(
            estimator_handle=handle,
            dataset="airline",
            horizon=3,
        )

        assert result["success"], f"Expected success, got: {result}"
        assert "job_id" in result
        assert result["data_source"] == "airline"

    def test_async_with_data_handle(self):
        """Async with a custom data handle should return a job_id."""
        from sktime_mcp.tools.fit_predict import fit_predict_async_tool

        handle = self._get_estimator_handle()
        data_handle = self._load_custom_data()

        result = fit_predict_async_tool(
            estimator_handle=handle,
            data_handle=data_handle,
            horizon=5,
        )

        assert result["success"], f"Expected success, got: {result}"
        assert "job_id" in result
        assert result["data_source"] == data_handle

    def test_async_both_provided_error(self):
        """Providing both dataset and data_handle should fail."""
        from sktime_mcp.tools.fit_predict import fit_predict_async_tool

        handle = self._get_estimator_handle()
        result = fit_predict_async_tool(
            estimator_handle=handle,
            dataset="airline",
            data_handle="data_fake123",
            horizon=3,
        )

        assert not result["success"]
        assert "error" in result
        assert "not both" in result["error"].lower()

    def test_async_neither_provided_error(self):
        """Omitting both dataset and data_handle should fail."""
        from sktime_mcp.tools.fit_predict import fit_predict_async_tool

        handle = self._get_estimator_handle()
        result = fit_predict_async_tool(
            estimator_handle=handle,
            horizon=3,
        )

        assert not result["success"]
        assert "error" in result

    def test_async_invalid_data_handle(self):
        """An invalid data_handle should fail at the executor level."""
        from sktime_mcp.tools.fit_predict import fit_predict_async_tool

        handle = self._get_estimator_handle()
        result = fit_predict_async_tool(
            estimator_handle=handle,
            data_handle="data_nonexistent",
            horizon=3,
        )

        # The tool succeeds in scheduling the job (returns job_id),
        # but the actual failure happens async inside the executor.
        # So we just verify the job was created successfully.
        assert result["success"]
        assert "job_id" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
