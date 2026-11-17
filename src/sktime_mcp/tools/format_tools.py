"""
Time series formatting tools for sktime MCP.

Provides tools for automatically fixing time series data format issues.
"""

from typing import Any

from sktime_mcp.runtime.executor import get_executor


def format_time_series_tool(
    data_handle: str,
    auto_infer_freq: bool = True,
    fill_missing: bool = True,
    remove_duplicates: bool = True,
) -> dict[str, Any]:
    """
    Automatically format time series data to be sktime-compatible.

    This tool fixes common issues:
    - Missing frequency on DatetimeIndex
    - Duplicate timestamps
    - Missing values
    - Irregular time intervals

    Args:
        data_handle: Handle from load_data_source
        auto_infer_freq: Automatically infer and set frequency (default: True)
        fill_missing: Fill missing values with forward/backward fill (default: True)
        remove_duplicates: Remove duplicate timestamps (default: True)

    Returns:
        Dictionary with:
        - success: bool
        - data_handle: str (new handle with formatted data)
        - changes_made: dict (what was fixed)
        - metadata: dict (updated metadata)

    Example:
        >>> format_time_series_tool(
        ...     data_handle="data_abc123",
        ...     auto_infer_freq=True,
        ...     fill_missing=True
        ... )
    """
    executor = get_executor()

    try:
        # Delegate to executor
        return executor.format_data_handle(
            data_handle,
            auto_infer_freq=auto_infer_freq,
            fill_missing=fill_missing,
            remove_duplicates=remove_duplicates,
        )

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__,
        }


def auto_format_on_load_tool(enabled: bool = True) -> dict[str, Any]:
    """
    Enable/disable automatic formatting when loading data.

    When enabled, all data loaded via load_data_source will be
    automatically formatted to be sktime-compatible.

    Args:
        enabled: Whether to enable auto-formatting (default: True)

    Returns:
        Dictionary with success status and current setting
    """
    executor = get_executor()

    # Store setting in executor
    if not hasattr(executor, "_auto_format_enabled"):
        executor._auto_format_enabled = True

    executor._auto_format_enabled = enabled

    return {
        "success": True,
        "auto_format_enabled": enabled,
        "message": f"Auto-formatting {'enabled' if enabled else 'disabled'}",
    }
