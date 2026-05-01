"""Tests for recommend_forecaster tool."""

import json

import pytest

from sktime_mcp.tools.recommend_forecaster import (
    recommend_forecaster,
    recommend_forecaster_tool,
)


class TestRecommendForecaster:
    """Test cases for recommend_forecaster function."""

    def test_basic_recommendation(self):
        """Test basic recommendation without parameters."""
        result = recommend_forecaster()
        assert "recommendations" in result
        assert "alternatives" in result
        assert "warnings" in result
        assert isinstance(result["recommendations"], list)
        assert isinstance(result["alternatives"], list)
        assert isinstance(result["warnings"], list)

    def test_limited_data_warning(self):
        """Test that limited data triggers appropriate warning."""
        data_chars = {"length": 20}
        result = recommend_forecaster(data_characteristics=data_chars)
        
        assert len(result["warnings"]) > 0
        assert any("Limited data" in w for w in result["warnings"])
        assert any(r["name"] == "NaiveForecaster" for r in result["recommendations"])

    def test_seasonal_and_trend_recommendation(self):
        """Test recommendation for data with seasonality and trend."""
        data_chars = {
            "length": 100,
            "seasonality": True,
            "trend": True,
        }
        requirements = {"accuracy": "high"}
        
        result = recommend_forecaster(
            data_characteristics=data_chars,
            requirements=requirements
        )
        
        assert len(result["recommendations"]) > 0
        assert any(r["name"] == "AutoARIMA" for r in result["recommendations"])

    def test_seasonal_only_recommendation(self):
        """Test recommendation for seasonal data without trend."""
        data_chars = {
            "length": 100,
            "seasonality": True,
            "trend": False,
        }
        
        result = recommend_forecaster(data_characteristics=data_chars)
        
        assert any(r["name"] == "SeasonalNaive" for r in result["recommendations"])

    def test_trend_only_recommendation(self):
        """Test recommendation for trending data without seasonality."""
        data_chars = {
            "length": 100,
            "seasonality": False,
            "trend": True,
        }
        
        result = recommend_forecaster(data_characteristics=data_chars)
        
        assert any(r["name"] == "TrendForecaster" for r in result["recommendations"])

    def test_multivariate_recommendation(self):
        """Test recommendation for multivariate data."""
        data_chars = {
            "length": 100,
            "multivariate": True,
        }
        
        result = recommend_forecaster(data_characteristics=data_chars)
        
        assert any(r["name"] == "VAR" for r in result["recommendations"])

    def test_missing_values_warning(self):
        """Test that missing values trigger appropriate warning."""
        data_chars = {
            "length": 100,
            "missing_values": True,
        }
        
        result = recommend_forecaster(data_characteristics=data_chars)
        
        assert any("missing values" in w.lower() for w in result["warnings"])

    def test_high_interpretability_warning(self):
        """Test that high interpretability requirement triggers warning."""
        requirements = {"interpretability": "high"}
        
        result = recommend_forecaster(requirements=requirements)
        
        assert any("interpretability" in w.lower() for w in result["warnings"])

    def test_conflicting_requirements_warning(self):
        """Test that conflicting requirements trigger warning."""
        requirements = {
            "speed": "high",
            "accuracy": "high",
        }
        
        result = recommend_forecaster(requirements=requirements)
        
        assert any("conflicting" in w.lower() for w in result["warnings"])

    def test_recommendation_structure(self):
        """Test that recommendations have proper structure."""
        data_chars = {"length": 100, "seasonality": True}
        result = recommend_forecaster(data_characteristics=data_chars)
        
        for rec in result["recommendations"]:
            assert "name" in rec
            assert "rationale" in rec
            assert isinstance(rec["name"], str)
            assert isinstance(rec["rationale"], str)


class TestRecommendForecasterTool:
    """Test cases for recommend_forecaster_tool MCP wrapper."""

    def test_tool_with_no_parameters(self):
        """Test tool wrapper with no parameters."""
        result = recommend_forecaster_tool()
        parsed = result
        
        assert "recommendations" in parsed
        assert "alternatives" in parsed
        assert "warnings" in parsed

    def test_tool_with_json_parameters(self):
        """Test tool wrapper with JSON string parameters."""
        data_chars_json = json.dumps({
            "length": 100,
            "seasonality": True,
            "trend": True,
        })
        requirements_json = json.dumps({
            "accuracy": "high",
        })
        
        result = recommend_forecaster_tool(
            data_characteristics=data_chars_json,
            requirements=requirements_json
        )
        parsed = result
        
        assert "recommendations" in parsed
        assert len(parsed["recommendations"]) > 0

    def test_tool_output_is_valid_json(self):
        """Test that tool output is valid JSON."""
        result = recommend_forecaster_tool()
        
        # Should not raise exception
        parsed = result
        assert isinstance(parsed, dict)

    def test_tool_with_empty_json_objects(self):
        """Test tool with empty JSON objects."""
        result = recommend_forecaster_tool(
            data_characteristics="{}",
            requirements="{}"
        )
        parsed = result
        
        assert "recommendations" in parsed
        assert "alternatives" in parsed
        assert "warnings" in parsed


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
