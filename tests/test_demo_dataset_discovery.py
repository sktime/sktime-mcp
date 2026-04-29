"""Tests for supported demo dataset discovery."""

import sys

sys.path.insert(0, "src")


def test_list_datasets_excludes_non_forecasting_supervised_demos():
    """Supervised `(X, y)` demo datasets should not be advertised as supported demos."""
    from sktime_mcp.runtime.executor import Executor

    executor = Executor()
    datasets = executor.list_datasets()

    assert "airline" in datasets
    assert "basic_motions" not in datasets
    assert "arrow_head" not in datasets
    assert "gunpoint" not in datasets


def test_list_available_data_demos_only_excludes_unsupported_datasets():
    """The MCP-facing demo discovery tool should inherit the filtered dataset list."""
    from sktime_mcp.tools.list_available_data import list_available_data_tool

    result = list_available_data_tool(is_demo=True)

    assert result["success"] is True
    assert "airline" in result["system_demos"]
    assert "basic_motions" not in result["system_demos"]
    assert "arrow_head" not in result["system_demos"]
    assert "gunpoint" not in result["system_demos"]
