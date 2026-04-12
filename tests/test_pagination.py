"""
Tests for pagination in list_available_data_tool and list_data_sources_tool.

Resolves: https://github.com/sktime/sktime-mcp/issues/153
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# ---------------------------------------------------------------------------
# list_available_data_tool tests
# ---------------------------------------------------------------------------


def test_list_available_data_returns_pagination_fields():
    """list_available_data_tool result must contain all pagination fields."""
    from sktime_mcp.tools.list_available_data import list_available_data_tool

    result = list_available_data_tool()

    assert result["success"] is True
    assert "total" in result
    assert "count" in result
    assert "offset" in result
    assert "limit" in result
    assert "has_more" in result


def test_list_available_data_default_pagination():
    """Default call should use offset=0 and limit=50."""
    from sktime_mcp.tools.list_available_data import list_available_data_tool

    result = list_available_data_tool()

    assert result["offset"] == 0
    assert result["limit"] == 50
    assert result["count"] <= 50


def test_list_available_data_limit():
    """Requesting limit=1 should return at most 1 item."""
    from sktime_mcp.tools.list_available_data import list_available_data_tool

    result = list_available_data_tool(limit=1)

    assert result["success"] is True
    total_in_page = len(result["system_demos"]) + len(result["active_handles"])
    assert total_in_page <= 1
    assert result["count"] <= 1


def test_list_available_data_offset_beyond_total():
    """Offset beyond total should return empty page with has_more=False."""
    from sktime_mcp.tools.list_available_data import list_available_data_tool

    result_all = list_available_data_tool()
    total = result_all["total"]

    result = list_available_data_tool(offset=total + 100)

    assert result["success"] is True
    assert result["count"] == 0
    assert result["has_more"] is False


def test_list_available_data_negative_offset_returns_error():
    """Negative offset must return an error response."""
    from sktime_mcp.tools.list_available_data import list_available_data_tool

    result = list_available_data_tool(offset=-1)

    assert result["success"] is False
    assert "error" in result


def test_list_available_data_pagination_consistency():
    """count field must equal the actual number of items returned in the page."""
    from sktime_mcp.tools.list_available_data import list_available_data_tool

    result = list_available_data_tool(limit=5, offset=0)

    assert result["success"] is True
    actual_count = len(result["system_demos"]) + len(result["active_handles"])
    assert result["count"] == actual_count


def test_list_available_data_has_more_flag():
    """has_more should be True when total > limit, False otherwise."""
    from sktime_mcp.tools.list_available_data import list_available_data_tool

    result = list_available_data_tool(limit=1, offset=0)
    total = result["total"]

    if total > 1:
        assert result["has_more"] is True
    else:
        assert result["has_more"] is False


# ---------------------------------------------------------------------------
# list_data_sources_tool tests
# ---------------------------------------------------------------------------


def test_list_data_sources_returns_pagination_fields():
    """list_data_sources_tool result must contain all pagination fields."""
    from sktime_mcp.tools.data_tools import list_data_sources_tool

    result = list_data_sources_tool()

    assert result["success"] is True
    assert "total" in result
    assert "count" in result
    assert "offset" in result
    assert "limit" in result
    assert "has_more" in result


def test_list_data_sources_default_pagination():
    """Default call should use offset=0 and limit=50."""
    from sktime_mcp.tools.data_tools import list_data_sources_tool

    result = list_data_sources_tool()

    assert result["offset"] == 0
    assert result["limit"] == 50
    assert result["count"] <= 50


def test_list_data_sources_limit():
    """Requesting limit=1 should return at most 1 source."""
    from sktime_mcp.tools.data_tools import list_data_sources_tool

    result = list_data_sources_tool(limit=1)

    assert result["success"] is True
    assert len(result["sources"]) <= 1
    assert result["count"] <= 1


def test_list_data_sources_offset_beyond_total():
    """Offset beyond total should return empty page with has_more=False."""
    from sktime_mcp.tools.data_tools import list_data_sources_tool

    result_all = list_data_sources_tool()
    total = result_all["total"]

    result = list_data_sources_tool(offset=total + 100)

    assert result["success"] is True
    assert result["count"] == 0
    assert len(result["sources"]) == 0
    assert result["has_more"] is False


def test_list_data_sources_negative_offset_returns_error():
    """Negative offset must return an error response."""
    from sktime_mcp.tools.data_tools import list_data_sources_tool

    result = list_data_sources_tool(offset=-1)

    assert result["success"] is False
    assert "error" in result


def test_list_data_sources_descriptions_match_sources():
    """Descriptions dict keys must match the sources list exactly."""
    from sktime_mcp.tools.data_tools import list_data_sources_tool

    result = list_data_sources_tool(limit=2, offset=0)

    assert result["success"] is True
    assert set(result["descriptions"].keys()) == set(result["sources"])


def test_list_data_sources_count_consistency():
    """count field must equal the actual number of sources returned."""
    from sktime_mcp.tools.data_tools import list_data_sources_tool

    result = list_data_sources_tool(limit=2, offset=0)

    assert result["success"] is True
    assert result["count"] == len(result["sources"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
