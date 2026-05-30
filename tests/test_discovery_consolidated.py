"""Tests for consolidated discovery tools: query_registry and describe_component."""

import pytest
from sktime_mcp.tools.list_estimators import query_registry_tool
from sktime_mcp.tools.describe_estimator import describe_component_tool


def test_query_registry_estimators():
    """Test query_registry for target='estimators'."""
    # Test getting estimators
    res = query_registry_tool(target="estimators", limit=10)
    assert res["success"]
    assert "results" in res
    assert len(res["results"]) > 0
    assert any(e["task"] == "forecasting" for e in res["results"])


def test_query_registry_tags():
    """Test query_registry for target='tags'."""
    # Test getting tags
    res = query_registry_tool(target="tags")
    assert res["success"]
    assert "results" in res
    assert len(res["results"]) > 0
    tag_names = {t["tag"] for t in res["results"]}
    assert "scitype:y" in tag_names or "capability:pred_int" in tag_names


def test_query_registry_metrics():
    """Test query_registry for target='metrics'."""
    # Test getting performance metrics
    res = query_registry_tool(target="metrics", limit=20)
    assert res["success"]
    assert "results" in res
    assert len(res["results"]) > 0
    # Metrics should have task == 'metric'
    assert any(e["task"] == "metric" for e in res["results"])


def test_describe_component_forecaster():
    """Test describe_component on a forecaster component."""
    res = describe_component_tool("NaiveForecaster")
    assert res["success"]
    assert res["name"] == "NaiveForecaster"
    assert res["task"] == "forecasting"
    assert "strategy" in res["parameters"]


def test_describe_component_metric():
    """Test describe_component on a metric component."""
    res = describe_component_tool("MeanAbsolutePercentageError")
    assert res["success"]
    assert res["name"] == "MeanAbsolutePercentageError"
    assert res["task"] == "metric"
