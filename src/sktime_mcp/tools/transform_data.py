"""Legacy Python helper. MCP clients use prepare_data with explicit policies."""

from sktime_mcp.runtime.executor import get_executor


def transform_data_tool(
    data_handle,
    action="format",
    auto_infer_freq=True,
    fill_missing=True,
    remove_duplicates=True,
    to_mtype=None,
):
    if action == "convert":
        return get_executor().prepare_data(data_handle, action, to_mtype=to_mtype)
    if action == "format":
        return get_executor().prepare_data(
            data_handle,
            action,
            sort_index=True,
            missing="ffill" if fill_missing else "keep",
            duplicates="first" if remove_duplicates else "keep",
        )
    return {"success": False, "error": "action must be format or convert"}
