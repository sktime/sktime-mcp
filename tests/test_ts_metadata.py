"""
Tests for extract_ts_metadata tool.
"""

import sys

import pytest

sys.path.insert(0, "src")


def test_extract_ts_metadata_demo_dataset():
    """Test extract_ts_metadata with a demo dataset (airline)."""
    from sktime_mcp.tools.ts_metadata import extract_ts_metadata_tool

    result = extract_ts_metadata_tool(dataset="airline")

    assert result["success"], f"Tool failed: {result.get('error')}"
    assert result["series_length"] == 144
    assert result["missing_count"] == 0
    assert result["missing_pct"] == 0.0
    assert result["mean"] is not None
    assert result["std"] is not None
    assert result["min"] is not None
    assert result["max"] is not None


def test_extract_ts_metadata_trend_detected():
    """Airline dataset has a known upward trend."""
    from sktime_mcp.tools.ts_metadata import extract_ts_metadata_tool

    result = extract_ts_metadata_tool(dataset="airline")

    assert result["success"]
    assert result["trend"] == "upward"


def test_extract_ts_metadata_stationarity():
    """Airline dataset is known to be non-stationary."""
    from sktime_mcp.tools.ts_metadata import extract_ts_metadata_tool

    result = extract_ts_metadata_tool(dataset="airline")

    assert result["success"]
    stationarity = result["stationarity"]
    assert "is_stationary" in stationarity
    assert stationarity["is_stationary"] is False
    assert "adf_statistic" in stationarity
    assert "adf_pvalue" in stationarity


def test_extract_ts_metadata_seasonality():
    """Airline dataset has strong monthly seasonality (period=12)."""
    from sktime_mcp.tools.ts_metadata import extract_ts_metadata_tool

    result = extract_ts_metadata_tool(dataset="airline")

    assert result["success"]
    seasonality = result["seasonality"]
    assert seasonality["detected"]
    assert seasonality["dominant_period"] == 12
    assert seasonality["strength"] in ("strong", "moderate")


def test_extract_ts_metadata_autocorrelation():
    """Autocorrelation keys are present."""
    from sktime_mcp.tools.ts_metadata import extract_ts_metadata_tool

    result = extract_ts_metadata_tool(dataset="airline")

    assert result["success"]
    acf = result["autocorrelation"]
    assert "lag_1" in acf
    assert "lag_seasonal" in acf
    assert acf["lag_1"] is not None


def test_extract_ts_metadata_recommended_models():
    """Recommended models list is non-empty and contains known models."""
    from sktime_mcp.tools.ts_metadata import extract_ts_metadata_tool

    result = extract_ts_metadata_tool(dataset="airline")

    assert result["success"]
    models = result["recommended_models"]
    assert isinstance(models, list)
    assert len(models) > 0
    # All recommendations should be strings (estimator names)
    assert all(isinstance(m, str) for m in models)


def test_extract_ts_metadata_agent_hints():
    """Agent hints list is non-empty and contains strings."""
    from sktime_mcp.tools.ts_metadata import extract_ts_metadata_tool

    result = extract_ts_metadata_tool(dataset="airline")

    assert result["success"]
    hints = result["agent_hints"]
    assert isinstance(hints, list)
    assert len(hints) > 0
    assert all(isinstance(h, str) for h in hints)


def test_extract_ts_metadata_with_data_handle():
    """Test extract_ts_metadata using a loaded data_handle."""
    from sktime.datasets import load_airline

    from sktime_mcp.runtime.executor import get_executor
    from sktime_mcp.tools.ts_metadata import extract_ts_metadata_tool

    executor = get_executor()
    y = load_airline()

    # Manually store a data handle
    handle_id = "data_test_meta"
    executor._data_handles[handle_id] = {
        "y": y,
        "X": None,
        "metadata": {"frequency": "MS"},
        "validation": {},
        "config": {},
    }

    try:
        result = extract_ts_metadata_tool(data_handle=handle_id)
        assert result["success"], f"Tool failed: {result.get('error')}"
        assert result["series_length"] == 144
        assert result["trend"] == "upward"
    finally:
        executor._data_handles.pop(handle_id, None)


def test_extract_ts_metadata_missing_both_args():
    """Should return error when neither data_handle nor dataset is provided."""
    from sktime_mcp.tools.ts_metadata import extract_ts_metadata_tool

    result = extract_ts_metadata_tool()

    assert result["success"] is False
    assert "error" in result


def test_extract_ts_metadata_unknown_handle():
    """Should return error for unknown data handle."""
    from sktime_mcp.tools.ts_metadata import extract_ts_metadata_tool

    result = extract_ts_metadata_tool(data_handle="data_nonexistent")

    assert result["success"] is False
    assert "error" in result


def test_extract_ts_metadata_lynx():
    """Test with lynx dataset — different characteristics from airline."""
    from sktime_mcp.tools.ts_metadata import extract_ts_metadata_tool

    result = extract_ts_metadata_tool(dataset="lynx")

    assert result["success"], f"Tool failed: {result.get('error')}"
    assert result["series_length"] > 0
    assert "trend" in result
    assert "stationarity" in result
    assert "seasonality" in result
    assert "recommended_models" in result
    assert "agent_hints" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
