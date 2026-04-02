from sktime_mcp.tools.data_tools import list_available_data_tool


def test_list_available_data_default():
    result = list_available_data_tool()
    assert result["success"] is True
    assert "system_demos" in result
    assert "active_handles" in result


def test_list_available_data_demos_only():
    result = list_available_data_tool(is_demo=True)
    assert "system_demos" in result
    assert "active_handles" not in result


def test_list_available_data_handles_only():
    result = list_available_data_tool(is_demo=False)
    assert "active_handles" in result
    assert "system_demos" not in result
