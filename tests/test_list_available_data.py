"""Tests for list_available_data tool validation."""

import sys

sys.path.insert(0, "src")


def test_list_available_data_rejects_string_is_demo():
    """Stringified booleans should not return misleading empty results."""
    from sktime_mcp.tools.list_available_data import list_available_data_tool

    result = list_available_data_tool("false")

    assert result["success"] is False
    assert result["error"] == "Invalid is_demo type 'str'. Expected a boolean value or None."


def test_list_available_data_rejects_numeric_is_demo():
    """Numeric values should be rejected instead of being treated as filters."""
    from sktime_mcp.tools.list_available_data import list_available_data_tool

    result = list_available_data_tool(1)

    assert result["success"] is False
    assert result["error"] == "Invalid is_demo type 'int'. Expected a boolean value or None."


def test_list_available_data_rejects_container_is_demo():
    """Container types should also fail validation cleanly."""
    from sktime_mcp.tools.list_available_data import list_available_data_tool

    result = list_available_data_tool([])

    assert result["success"] is False
    assert result["error"] == "Invalid is_demo type 'list'. Expected a boolean value or None."


def test_list_available_data_valid_none_still_returns_data():
    """The valid default path should remain unchanged."""
    from sktime_mcp.tools.list_available_data import list_available_data_tool

    result = list_available_data_tool()

    assert result["success"] is True
    assert "system_demos" in result
    assert "active_handles" in result
    assert "total" in result
