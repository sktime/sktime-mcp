"""Legacy Python split helper. MCP clients use prepare_data(action="split")."""

from sktime_mcp.runtime.executor import get_executor


def split_data_tool(data_handle, test_size=None, fh=None):
    return get_executor().prepare_data(data_handle, "split", test_size=test_size, fh=fh)
