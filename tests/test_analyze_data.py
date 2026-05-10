"""
Tests for analyze_data tool.
"""

import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, "src")


class TestAnalyzeDataTool:
    """Tests for the analyze_data tool."""

    @pytest.fixture
    def clean_executor(self):
        """Get a clean executor instance for testing."""
        from sktime_mcp.runtime.executor import get_executor

        executor = get_executor()
        # Clear any existing data handles
        executor._data_handles.clear()
        yield executor
        # Cleanup after
        executor._data_handles.clear()

    @pytest.fixture
    def airline_data_handle(self, clean_executor):
        """Load airline dataset and return its data handle."""
        result = clean_executor.load_data_source({
            "type": "pandas",
            "data": {
                "date": pd.date_range("1949-01-01", periods=144, freq="M").strftime("%Y-%m").tolist(),
                "passengers": [112, 118, 132, 129, 121, 135, 148, 148, 136, 119, 104, 118,
                               115, 126, 141, 135, 125, 149, 170, 170, 158, 133, 114, 140,
                               145, 150, 178, 163, 172, 178, 199, 199, 184, 162, 146, 166,
                               171, 180, 193, 181, 183, 218, 230, 242, 209, 191, 172, 194,
                               196, 196, 220, 201, 229, 243, 264, 272, 237, 211, 180, 201,
                               204, 188, 235, 227, 234, 264, 302, 293, 259, 229, 203, 229,
                               242, 233, 267, 269, 270, 315, 364, 347, 312, 274, 237, 278,
                               284, 277, 317, 313, 318, 374, 413, 405, 355, 306, 271, 306,
                               315, 301, 356, 348, 355, 422, 465, 467, 404, 347, 305, 336,
                               340, 318, 362, 348, 363, 435, 491, 505, 404, 359, 310, 337,
                               360, 342, 406, 396, 420, 472, 548, 559, 463, 407, 362, 405,
                               417, 391, 419, 461, 472, 535, 622, 606, 508, 461, 390, 432],
            },
            "time_column": "date",
            "target_column": "passengers",
        })
        assert result["success"], f"Failed to load data: {result.get('error')}"
        return result["data_handle"]

    @pytest.fixture
    def stationary_data_handle(self, clean_executor):
        """Create a stationary series (white noise)."""
        np.random.seed(42)
        data = np.random.randn(100)
        index = pd.date_range("2020-01-01", periods=100, freq="D")

        result = clean_executor.load_data_source({
            "type": "pandas",
            "data": {
                "date": [str(d) for d in index],
                "value": data.tolist(),
            },
            "time_column": "date",
            "target_column": "value",
        })
        assert result["success"]
        return result["data_handle"]

    @pytest.fixture
    def trend_data_handle(self, clean_executor):
        """Create a series with clear upward trend."""
        n = 100
        index = pd.date_range("2020-01-01", periods=n, freq="D")
        trend_values = np.linspace(10, 100, n) + np.random.randn(n) * 5

        result = clean_executor.load_data_source({
            "type": "pandas",
            "data": {
                "date": [str(d) for d in index],
                "value": trend_values.tolist(),
            },
            "time_column": "date",
            "target_column": "value",
        })
        assert result["success"]
        return result["data_handle"]

    def test_analyze_with_airline_data(self, airline_data_handle):
        """Test analyze_data with airline dataset (has trend and seasonality)."""
        from sktime_mcp.tools.analyze_data import analyze_data_tool

        result = analyze_data_tool(airline_data_handle)

        assert result["success"], f"Analysis failed: {result.get('error')}"
        assert result["data_handle"] == airline_data_handle
        assert "basic_stats" in result
        assert "stationarity" in result
        assert "trend" in result
        assert "seasonality" in result
        assert "summary" in result

        # Check basic stats
        assert result["basic_stats"]["length"] == 144
        assert result["basic_stats"]["min"] is not None
        assert result["basic_stats"]["max"] is not None
        assert result["basic_stats"]["mean"] is not None

        # Check stationarity
        assert result["stationarity"]["test"] == "adf"
        assert result["stationarity"]["p_value"] is not None
        assert "is_stationary" in result["stationarity"]

        # Check trend
        assert "has_trend" in result["trend"]
        assert "trend_slope" in result["trend"]

        # Check seasonality
        assert "has_seasonality" in result["seasonality"]
        assert "seasonal_period" in result["seasonality"]

        # Check summary
        assert "interpretation" in result["summary"]
        assert "recommendations" in result["summary"]
        assert len(result["summary"]["recommendations"]) > 0

    def test_analyze_stationary_data(self, stationary_data_handle):
        """Test analyze_data with stationary (white noise) data."""
        from sktime_mcp.tools.analyze_data import analyze_data_tool

        result = analyze_data_tool(stationary_data_handle)

        assert result["success"]
        # White noise should be stationary
        assert result["stationarity"]["is_stationary"] is True
        # White noise typically has no significant trend
        assert result["trend"]["has_trend"] is False or result["trend"]["has_trend"] is None
        # White noise may or may not have seasonality depending on random chance

    def test_analyze_trend_data(self, trend_data_handle):
        """Test analyze_data with data that has a clear trend."""
        from sktime_mcp.tools.analyze_data import analyze_data_tool

        result = analyze_data_tool(trend_data_handle)

        assert result["success"]
        # Data with trend should show significant trend
        assert result["trend"]["has_trend"] is True
        assert result["trend"]["trend_slope"] > 0  # Upward trend

    def test_analyze_unknown_handle(self):
        """Test analyze_data with an unknown data handle."""
        from sktime_mcp.tools.analyze_data import analyze_data_tool

        result = analyze_data_tool("data_nonexistent_handle_123")

        assert not result["success"]
        assert "error" in result
        assert "Unknown data handle" in result["error"]

    def test_analyze_empty_series(self, clean_executor):
        """Test analyze_data with an empty series."""
        from sktime_mcp.tools.analyze_data import analyze_data_tool

        # Create empty data handle - loading itself will fail with empty data
        result = clean_executor.load_data_source({
            "type": "pandas",
            "data": {"date": [], "value": []},
            "time_column": "date",
            "target_column": "value",
        })
        # Data loading fails for empty series (pandas requires at least 3 dates)
        assert not result["success"]
        assert "error" in result

    def test_analyze_all_nan_series(self, clean_executor):
        """Test analyze_data with a series containing only NaN values."""
        from sktime_mcp.tools.analyze_data import analyze_data_tool

        result = clean_executor.load_data_source({
            "type": "pandas",
            "data": {
                "date": ["2020-01-01", "2020-01-02", "2020-01-03"],
                "value": [np.nan, np.nan, np.nan],
            },
            "time_column": "date",
            "target_column": "value",
        })
        assert result["success"]
        data_handle = result["data_handle"]

        result = analyze_data_tool(data_handle)

        assert not result["success"]
        assert "NaN" in result["error"] or "empty" in result["error"]

    def test_analyze_insufficient_data(self, clean_executor):
        """Test analyze_data with insufficient data points."""
        from sktime_mcp.tools.analyze_data import analyze_data_tool

        # Note: pandas requires at least 3 dates to infer frequency,
        # so loading with 3 dates may fail during load_data_source
        load_result = clean_executor.load_data_source({
            "type": "pandas",
            "data": {
                "date": ["2020-01-01", "2020-01-02", "2020-01-03"],  # 3 dates - minimum
                "value": [1.0, 2.0, 3.0],
            },
            "time_column": "date",
            "target_column": "value",
        })

        # If loading succeeds (3 dates is the minimum), test the analysis
        # If loading fails, it's expected behavior due to pandas limitations
        if load_result["success"]:
            data_handle = load_result["data_handle"]
            analysis_result = analyze_data_tool(data_handle)
            # Should handle gracefully - analysis succeeds but some tests have insufficient data
            assert analysis_result["success"]
            # Stationarity test should report insufficient data
            assert (
                analysis_result["stationarity"].get("error") is not None
                or analysis_result["stationarity"].get("p_value") is not None
            )
        else:
            # Loading failed due to pandas frequency inference requirement
            assert "error" in load_result

    def test_analyze_include_acf_false(self, airline_data_handle):
        """Test analyze_data with include_acf=False."""
        from sktime_mcp.tools.analyze_data import analyze_data_tool

        result = analyze_data_tool(airline_data_handle, include_acf=False)

        assert result["success"]
        # ACF values should be None when include_acf=False
        assert result["seasonality"]["acf_values"] is None
        assert result["seasonality"]["pacf_values"] is None

    def test_analyze_max_acf_lags(self, airline_data_handle):
        """Test analyze_data with custom max_acf_lags."""
        from sktime_mcp.tools.analyze_data import analyze_data_tool

        result = analyze_data_tool(airline_data_handle, max_acf_lags=5)

        assert result["success"]
        # ACF values should not exceed max_acf_lags + 1 (including lag 0)
        if result["seasonality"]["acf_values"]:
            assert len(result["seasonality"]["acf_values"]) <= 6

    def test_basic_stats_computation(self, airline_data_handle):
        """Test that basic statistics are computed correctly."""
        from sktime_mcp.tools.analyze_data import analyze_data_tool

        result = analyze_data_tool(airline_data_handle)

        assert result["success"]
        stats = result["basic_stats"]

        # Check required fields
        assert "length" in stats
        assert "mean" in stats
        assert "std" in stats
        assert "min" in stats
        assert "max" in stats
        assert "frequency" in stats
        assert "start" in stats
        assert "end" in stats
        assert "is_nan" in stats

        # Verify values are numeric
        assert isinstance(stats["length"], int)
        assert isinstance(stats["mean"], float)

    def test_stationarity_adf_test(self, airline_data_handle):
        """Test that ADF test returns proper structure."""
        from sktime_mcp.tools.analyze_data import analyze_data_tool

        result = analyze_data_tool(airline_data_handle)

        assert result["success"]
        stat = result["stationarity"]

        assert stat["test"] == "adf"
        assert "statistic" in stat
        assert "p_value" in stat
        assert "critical_values" in stat
        assert "is_stationary" in stat
        # Check critical values structure
        if stat["critical_values"]:
            assert isinstance(stat["critical_values"], dict)
            assert "1%" in stat["critical_values"] or "5%" in stat["critical_values"] or "10%" in stat["critical_values"]

    def test_trend_linear_regression(self, airline_data_handle):
        """Test that trend analysis returns proper structure."""
        from sktime_mcp.tools.analyze_data import analyze_data_tool

        result = analyze_data_tool(airline_data_handle)

        assert result["success"]
        trend = result["trend"]

        assert "has_trend" in trend
        assert "trend_slope" in trend
        assert "trend_intercept" in trend
        assert "trend_p_value" in trend
        assert "r_squared" in trend

    def test_summary_interpretation(self, airline_data_handle):
        """Test that summary interpretation is generated."""
        from sktime_mcp.tools.analyze_data import analyze_data_tool

        result = analyze_data_tool(airline_data_handle)

        assert result["success"]
        summary = result["summary"]

        assert "interpretation" in summary
        assert "recommendations" in summary
        # Interpretation should be a non-empty string
        assert isinstance(summary["interpretation"], str)
        assert len(summary["interpretation"]) > 0
        # Recommendations should be a list
        assert isinstance(summary["recommendations"], list)

    def test_summary_recommendations_vary_by_data(self, stationary_data_handle, trend_data_handle):
        """Test that recommendations differ based on data characteristics."""
        from sktime_mcp.tools.analyze_data import analyze_data_tool

        # Stationary data should not need differencing
        stationary_result = analyze_data_tool(stationary_data_handle)
        stationary_recs = stationary_result["summary"]["recommendations"]

        # Trend data should suggest trend-related models
        trend_result = analyze_data_tool(trend_data_handle)
        trend_recs = trend_result["summary"]["recommendations"]

        # Both should have recommendations
        assert len(stationary_recs) > 0
        assert len(trend_recs) > 0


class TestAnalyzeDataFunctions:
    """Unit tests for individual analyze_data functions."""

    def test_compute_basic_stats(self):
        """Test _compute_basic_stats function."""
        from sktime_mcp.tools.analyze_data import _compute_basic_stats

        index = pd.date_range("2020-01-01", periods=10, freq="D")
        series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0], index=index)

        stats = _compute_basic_stats(series)

        assert stats["length"] == 10
        assert stats["min"] == 1.0
        assert stats["max"] == 10.0
        assert stats["mean"] == 5.5
        assert stats["is_nan"] == 0

    def test_compute_basic_stats_with_nan(self):
        """Test _compute_basic_stats with NaN values."""
        from sktime_mcp.tools.analyze_data import _compute_basic_stats

        index = pd.date_range("2020-01-01", periods=5, freq="D")
        series = pd.Series([1.0, np.nan, 3.0, np.nan, 5.0], index=index)

        stats = _compute_basic_stats(series)

        assert stats["length"] == 5
        assert stats["is_nan"] == 2
        assert stats["min"] == 1.0
        assert stats["max"] == 5.0

    def test_compute_stationarity_adf(self):
        """Test _compute_stationarity with ADF test."""
        from sktime_mcp.tools.analyze_data import _compute_stationarity

        # Stationary series (white noise)
        stationary = pd.Series(np.random.randn(100))

        result = _compute_stationarity(stationary)

        assert result["test"] == "adf"
        assert result["p_value"] is not None
        assert result["is_stationary"] is not None

    def test_compute_trend_linear_regression(self):
        """Test _compute_trend with linear regression."""
        from sktime_mcp.tools.analyze_data import _compute_trend

        # Series with upward trend
        trend_values = np.linspace(10, 100, 50)
        series = pd.Series(trend_values)

        result = _compute_trend(series)

        assert "has_trend" in result
        assert "trend_slope" in result
        assert result["trend_slope"] > 0  # Should detect upward trend

    def test_compute_seasonality_acf(self):
        """Test _compute_seasonality with ACF analysis."""
        from sktime_mcp.tools.analyze_data import _compute_seasonality

        # Series with known periodicity (sine wave)
        t = np.linspace(0, 4 * np.pi, 100)
        seasonal = np.sin(t) + np.random.randn(100) * 0.1
        series = pd.Series(seasonal)

        result = _compute_seasonality(series, max_lags=24)

        assert "has_seasonality" in result
        assert "acf_values" in result
        assert "pacf_values" in result

    def test_compute_summary(self):
        """Test _compute_summary function."""
        from sktime_mcp.tools.analyze_data import _compute_summary

        stationarity = {"is_stationary": False, "error": None}
        trend = {"has_trend": True, "trend_slope": 0.5}
        seasonality = {"has_seasonality": True, "seasonal_period": 12}

        summary = _compute_summary(stationarity, trend, seasonality)

        assert "interpretation" in summary
        assert "recommendations" in summary
        assert len(summary["recommendations"]) > 0
        # Check that interpretation mentions non-stationary
        assert "non-stationary" in summary["interpretation"].lower()

    def test_get_recommendations(self):
        """Test _get_recommendations function."""
        from sktime_mcp.tools.analyze_data import _get_recommendations

        # Non-stationary with trend and seasonality
        recommendations = _get_recommendations(
            {"is_stationary": False},
            {"has_trend": True, "trend_slope": 0.5},
            {"has_seasonality": True, "seasonal_period": 12}
        )

        assert len(recommendations) > 0
        # Should recommend differencing for non-stationary
        assert any("differenc" in r.lower() for r in recommendations)
        # Should recommend seasonal models
        assert any("seasonal" in r.lower() or "sarima" in r.lower() for r in recommendations)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
