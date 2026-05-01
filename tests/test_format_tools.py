"""Tests for formatting tool input validation."""

import sys

import pandas as pd

sys.path.insert(0, "src")


def test_format_time_series_rejects_non_boolean_control_flags():
    """Truthiness-based string flags should be rejected before formatting starts."""
    from sktime_mcp.runtime.executor import get_executor
    from sktime_mcp.tools.format_tools import format_time_series_tool

    executor = get_executor()
    data_handle = "data_format_validation"
    executor._data_handles[data_handle] = {
        "y": pd.Series(
            [1, None, 3], index=pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-02"])
        ),
        "X": None,
        "metadata": {},
        "validation": {},
        "config": {},
    }

    try:
        result = format_time_series_tool(
            data_handle,
            auto_infer_freq=True,
            fill_missing=False,
            remove_duplicates="false",
        )
    finally:
        executor._data_handles.pop(data_handle, None)

    assert result["success"] is False
    assert result["error"] == "Invalid remove_duplicates type 'str'. Expected a boolean value."


def test_format_time_series_rejects_non_boolean_fill_missing():
    """Non-boolean fill_missing should not silently trigger data filling."""
    from sktime_mcp.runtime.executor import get_executor
    from sktime_mcp.tools.format_tools import format_time_series_tool

    executor = get_executor()
    data_handle = "data_fill_validation"
    executor._data_handles[data_handle] = {
        "y": pd.Series(
            [1, None, 3], index=pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"])
        ),
        "X": None,
        "metadata": {},
        "validation": {},
        "config": {},
    }

    try:
        result = format_time_series_tool(
            data_handle,
            auto_infer_freq=False,
            fill_missing="false",
            remove_duplicates=False,
        )
    finally:
        executor._data_handles.pop(data_handle, None)

    assert result["success"] is False
    assert result["error"] == "Invalid fill_missing type 'str'. Expected a boolean value."


def test_format_time_series_rejects_non_boolean_auto_infer_freq():
    """Non-boolean auto_infer_freq should fail before inference logic runs."""
    from sktime_mcp.runtime.executor import get_executor
    from sktime_mcp.tools.format_tools import format_time_series_tool

    executor = get_executor()
    data_handle = "data_freq_validation"
    executor._data_handles[data_handle] = {
        "y": pd.Series([1, 2], index=pd.to_datetime(["2024-01-01", "2024-01-03"])),
        "X": None,
        "metadata": {},
        "validation": {},
        "config": {},
    }

    try:
        result = format_time_series_tool(
            data_handle,
            auto_infer_freq="false",
            fill_missing=False,
            remove_duplicates=False,
        )
    finally:
        executor._data_handles.pop(data_handle, None)

    assert result["success"] is False
    assert result["error"] == "Invalid auto_infer_freq type 'str'. Expected a boolean value."


def test_auto_format_on_load_rejects_non_boolean_enabled():
    """auto_format_on_load should reject stringified booleans."""
    from sktime_mcp.tools.format_tools import auto_format_on_load_tool

    result = auto_format_on_load_tool("false")

    assert result["success"] is False
    assert result["error"] == "Invalid enabled type 'str'. Expected a boolean value."
