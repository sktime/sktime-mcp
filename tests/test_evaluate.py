"""
Tests for evaluate tool.
"""

import sys

import pytest

sys.path.insert(0, "src")


def test_evaluate_estimator_tool():
    """Test evaluate_estimator_tool with a simple estimator."""
    from sktime_mcp.tools.evaluate import evaluate_estimator_tool
    from sktime_mcp.runtime.executor import get_executor
    from sktime.forecasting.naive import NaiveForecaster

    executor = get_executor()

    # Create handle manually for the test
    handle = executor._handle_manager.create_handle("NaiveForecaster", NaiveForecaster(), {})

    try:
        result = evaluate_estimator_tool(handle, "airline", cv_folds=2)

        assert result["success"], f"Evaluate failed: {result.get('error')}"
        assert "results" in result
        assert result["cv_folds_run"] == 2
        assert len(result["results"]) == 2
        
        # Verify result contains common metrics like test_MeanAbsolutePercentageError
        metric_columns = [k for k in result["results"][0].keys() if "test_" in k]
        assert len(metric_columns) > 0

    finally:
        # Clean up
        executor._handle_manager.release_handle(handle)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
