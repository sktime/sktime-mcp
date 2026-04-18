"""Tests for recommend_estimators MCP tool."""

import sys

sys.path.insert(0, "src")

from sktime_mcp.tools.recommend_estimators import recommend_estimators_tool


class TestRecommendEstimatorsTool:
    """Behavior tests for recommendation workflow."""

    def test_invalid_limit_rejected(self):
        """limit must be a positive integer."""
        result = recommend_estimators_tool(limit=0)
        assert result["success"] is False
        assert "limit" in result["error"]

    def test_query_infers_forecasting_constraints(self):
        """Free-text query should infer common forecasting constraints."""
        result = recommend_estimators_tool(
            query="Need monthly forecasting with prediction intervals and missing values",
            limit=3,
        )

        assert result["success"] is True
        assert result["inferred_from_query"]["task"] == "forecasting"
        assert result["inferred_from_query"]["required_tags"].get("capability:pred_int") is True
        assert result["count"] <= 3

    def test_required_tags_are_applied(self):
        """All returned recommendations must satisfy hard tag filters."""
        result = recommend_estimators_tool(
            task="forecasting",
            required_tags={"capability:pred_int": True},
            limit=5,
        )

        assert result["success"] is True
        for rec in result["recommendations"]:
            assert rec["task"] == "forecasting"
            assert rec["tags"].get("capability:pred_int") is True

    def test_scores_are_sorted_descending(self):
        """Returned recommendations should be sorted by score descending."""
        result = recommend_estimators_tool(
            query="forecast with intervals",
            task="forecasting",
            preferred_tags={"capability:pred_int": True},
            limit=5,
        )

        assert result["success"] is True
        scores = [rec["score"] for rec in result["recommendations"]]
        assert scores == sorted(scores, reverse=True)
