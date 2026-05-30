"""Tests for consolidated discovery tools: query_registry and describe_component."""

from sktime_mcp.tools.describe_component import describe_component_tool
from sktime_mcp.tools.list_estimators import query_registry_tool


def test_query_registry_all():
    """Test query_registry without filters."""
    res = query_registry_tool(limit=10)
    assert res["success"]
    assert "results" in res
    assert len(res["results"]) > 0
    assert any(e["task"] == "forecaster" for e in res["results"])


def test_query_registry_by_task():
    """Test query_registry filtered by scitype."""
    res = query_registry_tool(task="forecaster", limit=10)
    assert res["success"]
    assert all(e["task"] == "forecaster" for e in res["results"])


def test_query_registry_metrics():
    """Test query_registry can find metrics via task='metric'."""
    res = query_registry_tool(task="metric", limit=20)
    assert res["success"]
    assert "results" in res
    assert len(res["results"]) > 0
    assert all(e["task"] == "metric" for e in res["results"])


def test_query_registry_tags():
    """Test query_registry with task='tag' returns tag metadata."""
    res = query_registry_tool(task="tag")
    assert res["success"]
    assert "results" in res
    assert len(res["results"]) > 0
    tag_names = {t["tag"] for t in res["results"]}
    assert "scitype:y" in tag_names or "capability:pred_int" in tag_names


def test_describe_component_forecaster():
    """Test describe_component on a forecaster component."""
    res = describe_component_tool("NaiveForecaster")
    assert res["success"]
    assert res["name"] == "NaiveForecaster"
    assert res["task"] == "forecaster"
    assert "strategy" in res["parameters"]
    assert res["parameters"] == res["hyperparameters"]


def test_describe_component_metric():
    """Test describe_component on a metric component."""
    res = describe_component_tool("MeanAbsolutePercentageError")
    assert res["success"]
    assert res["name"] == "MeanAbsolutePercentageError"
    assert res["task"] == "metric"
    assert "parameters" in res
    assert res["parameters"] == res["hyperparameters"]
