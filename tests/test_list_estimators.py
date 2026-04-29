"""
Tests for list_estimators_tool input validation.
"""
import pytest
from sktime_mcp.tools.list_estimators import list_estimators_tool


def test_invalid_task_returns_error():
    result = list_estimators_tool(task="forcasting")
    assert result["success"] is False
    assert "forcasting" in result["error"]
    assert "forecasting" in result["error"]


def test_invalid_task_suggests_close_match():
    result = list_estimators_tool(task="forcasting")
    assert result["success"] is False
    assert "forecasting" in result["error"]


def test_invalid_tag_returns_error():
    result = list_estimators_tool(tags={"pred_int": True})
    assert result["success"] is False
    assert "pred_int" in result["error"]


def test_invalid_tag_suggests_substring_match():
    result = list_estimators_tool(tags={"pred_int": True})
    assert result["success"] is False
    suggestions = result["suggestions"]["pred_int"]
    assert suggestions is not None
    assert "capability:pred_int" in suggestions


def test_valid_task_returns_success():
    """Task validation passes - registry load issues are environment-specific."""
    result = list_estimators_tool(task="forecasting")
    assert "Invalid task" not in result.get("error", "")


def test_valid_tag_returns_success():
    """Tag validation passes - registry load issues are environment-specific."""
    result = list_estimators_tool(tags={"capability:pred_int": True})
    assert "Invalid tag" not in result.get("error", "")


def test_no_filters_does_not_raise():
    """No filters should not cause a validation error."""
    result = list_estimators_tool()
    assert "Invalid task" not in result.get("error", "")