"""Tests for diagnostic MCP tools (detect_seasonality, check_structural_break).

See: https://github.com/sktime/sktime-mcp/issues/96
"""

import math

import numpy as np
import pytest

from sktime_mcp.tools.diagnostics import (
    check_structural_break_tool,
    detect_seasonality_tool,
)


class TestDetectSeasonality:
    """Tests for the detect_seasonality tool."""

    def test_strong_weekly_seasonality(self):
        """A clear period-7 sine wave should be detected as strong seasonality."""
        np.random.seed(42)
        t = np.arange(200)
        data = np.sin(2 * np.pi * t / 7) + np.random.randn(200) * 0.1
        result = detect_seasonality_tool(data.tolist())

        assert result["success"] is True
        assert result["period"] == 7
        assert result["seasonality_class"] in ("strong", "moderate")
        assert result["confidence"] in ("high", "medium")
        assert len(result["candidates"]) > 0
        assert result["next_action_hint"] == "deseasonalize_then_stationarity"

    def test_no_seasonality_white_noise(self):
        """White noise should show no significant seasonality."""
        np.random.seed(42)
        data = np.random.randn(200).tolist()
        result = detect_seasonality_tool(data)

        assert result["success"] is True
        assert result["seasonality_class"] in ("none", "weak")

    def test_frequency_hint_boosts_known_periods(self):
        """Passing frequency='D' should boost daily-seasonal periods."""
        np.random.seed(42)
        t = np.arange(200)
        # Period-7 signal (weekly for daily data)
        data = (np.sin(2 * np.pi * t / 7) + np.random.randn(200) * 0.3).tolist()

        result_no_freq = detect_seasonality_tool(data)
        result_with_freq = detect_seasonality_tool(data, frequency="D")

        assert result_no_freq["success"] is True
        assert result_with_freq["success"] is True
        # Both should detect period 7
        assert result_with_freq["period"] == 7

    def test_short_series_returns_error(self):
        """Series with < 6 points should return an error."""
        result = detect_seasonality_tool([1.0, 2.0, 3.0])
        assert result["success"] is False
        assert "too short" in result["error"].lower()

    def test_custom_max_lag(self):
        """max_lag parameter should limit the search range."""
        np.random.seed(42)
        t = np.arange(200)
        # Period-7 signal with enough lag range to detect it
        data = (np.sin(2 * np.pi * t / 7) + np.random.randn(200) * 0.1).tolist()

        result_short = detect_seasonality_tool(data, max_lag=3)
        result_long = detect_seasonality_tool(data, max_lag=50)

        assert result_short["success"] is True
        assert result_long["success"] is True
        # Short lag range (max_lag=3) can't detect period-7
        # Long lag range should detect it
        assert result_long["period"] == 7

    def test_output_structure(self):
        """Verify all expected keys are present in the output."""
        np.random.seed(42)
        data = np.random.randn(100).tolist()
        result = detect_seasonality_tool(data)

        assert "success" in result
        assert "period" in result
        assert "strength" in result
        assert "seasonality_class" in result
        assert "confidence" in result
        assert "candidates" in result
        assert "method" in result
        assert "next_action_hint" in result


class TestCheckStructuralBreak:
    """Tests for the check_structural_break tool."""

    def test_clear_level_shift(self):
        """A series with a large level shift should be detected."""
        np.random.seed(42)
        before = np.random.randn(100) + 0.0
        after = np.random.randn(100) + 10.0  # Big jump at index 100
        data = np.concatenate([before, after]).tolist()
        result = check_structural_break_tool(data)

        assert result["success"] is True
        assert result["break_detected"] is True
        assert result["location"] is not None
        # Break should be near index 100 (±20)
        assert abs(result["location"] - 100) < 20
        assert result["next_action_hint"] == "get_dataset_history"

    def test_no_break_stationary(self):
        """A stationary series should have no structural break."""
        np.random.seed(42)
        data = np.random.randn(200).tolist()
        result = check_structural_break_tool(data)

        assert result["success"] is True
        assert result["break_detected"] is False
        assert result["next_action_hint"] == "proceed_with_full_series"

    def test_constant_series(self):
        """A constant series (zero std) should return no break."""
        data = [5.0] * 50
        result = check_structural_break_tool(data)

        assert result["success"] is True
        assert result["break_detected"] is False
        assert result["next_action_hint"] == "constant_series"

    def test_short_series_returns_error(self):
        """Series with < 10 points should return an error."""
        result = check_structural_break_tool([1.0, 2.0, 3.0])
        assert result["success"] is False
        assert "too short" in result["error"].lower()

    def test_output_structure(self):
        """Verify all expected keys are present in the output."""
        np.random.seed(42)
        data = np.random.randn(100).tolist()
        result = check_structural_break_tool(data)

        assert "success" in result
        assert "break_detected" in result
        assert "location" in result
        assert "confidence" in result
        assert "test_stat" in result
        assert "next_action_hint" in result

    def test_confidence_bounded(self):
        """Confidence should be between 0 and 1."""
        np.random.seed(42)
        data = np.random.randn(200).tolist()
        result = check_structural_break_tool(data)

        assert 0.0 <= result["confidence"] <= 1.0
