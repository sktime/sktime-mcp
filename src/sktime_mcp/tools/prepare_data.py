"""One explicit entry point for formatting, conversion and splitting."""

from typing import Any

from sktime_mcp.runtime.executor import get_executor


def prepare_data_tool(data_handle: str, action: str, **options) -> dict[str, Any]:
    """Prepare data without modifying the source; validate options by action."""
    return get_executor().prepare_data(data_handle, action, **options)
