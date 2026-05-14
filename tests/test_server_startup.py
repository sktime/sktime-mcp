"""
Tests to ensure the MCP server starts up correctly and all tool imports are valid.
This protects against regressions where tool exports are renamed or removed
but the server module is not updated.
"""

import asyncio

import pytest


def test_server_import_success():
    """Test that importing the server module succeeds without any ImportErrors."""
    import sktime_mcp.server as mcp_server

    assert hasattr(mcp_server, "server")
    assert mcp_server.server.name == "sktime-mcp"


def test_server_tools_registration():
    """Test that the list_tools endpoint returns expected tools."""
    import sktime_mcp.server as mcp_server

    tools = asyncio.run(mcp_server.list_tools())
    assert len(tools) > 0, "Server should register at least one tool"

    tool_names = [tool.name for tool in tools]
    expected_tools = [
        "list_estimators",
        "instantiate_estimator",
        "fit_predict",
        "load_data_source",
    ]
    for expected in expected_tools:
        assert expected in tool_names, f"Expected tool {expected} not found in registered tools"
