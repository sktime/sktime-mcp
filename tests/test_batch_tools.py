"""Tests for batch execution tool."""

import sys

sys.path.insert(0, "src")

import sktime_mcp.tools.batch as batch_tools


def test_run_tools_batch_success(monkeypatch):
    """run_tools_batch should execute supported read-only tools."""
    monkeypatch.setattr(batch_tools, "get_available_tags", lambda: {"success": True, "tags": []})
    monkeypatch.setattr(
        batch_tools,
        "list_estimators_tool",
        lambda **kwargs: {"success": True, "estimators": [], "kwargs": kwargs},
    )
    monkeypatch.setattr(
        batch_tools,
        "list_available_data_tool",
        lambda *args, **kwargs: {"success": True, "system_demos": ["airline"]},
    )

    result = batch_tools.run_tools_batch_tool(
        [
            {"tool": "get_available_tags"},
            {"tool": "list_estimators", "arguments": {"task": "forecasting", "limit": 2}},
            {"tool": "list_available_data"},
        ]
    )

    assert result["success"] is True
    assert result["count"] == 3
    assert all(item["success"] for item in result["results"])
    assert result["results"][1]["result"]["success"] is True


def test_run_tools_batch_rejects_unsupported_tool():
    """Unsupported tools should fail with clear per-operation errors."""
    result = batch_tools.run_tools_batch_tool(
        [
            {"tool": "release_handle", "arguments": {"handle": "est_123"}},
        ]
    )

    assert result["success"] is False
    assert result["results"][0]["success"] is False
    assert "Unsupported tool for batch execution" in result["results"][0]["error"]


def test_run_tools_batch_rejects_malformed_operation():
    """Malformed operations should be reported without crashing."""
    result = batch_tools.run_tools_batch_tool(
        [
            {"tool": "", "arguments": {}},
            {"tool": "list_estimators", "arguments": "not-a-dict"},
            "not-an-object",
        ]
    )

    assert result["success"] is False
    assert result["count"] == 3
    assert all(item["success"] is False for item in result["results"])

