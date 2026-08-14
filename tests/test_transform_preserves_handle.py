"""transform_data(action='format') must not destroy the caller's input handle.

The tool contract is "transform a loaded data handle and return a NEW
handle" — the input must stay valid, exactly as action='convert' already
behaves. Only internal auto-format-on-load may release the raw handle,
because that handle was never exposed to the caller.
"""

import pandas as pd
import pytest

from sktime_mcp.runtime.executor import get_executor
from sktime_mcp.tools.transform_data import transform_data_tool


@pytest.fixture
def data_handle():
    executor = get_executor()
    idx = pd.date_range("2024-01-01", periods=10, freq="D")
    executor._data_handles["test_fmt_parent"] = {
        "y": pd.Series(range(10), index=idx, dtype=float),
        "X": None,
        "metadata": {"rows": 10, "frequency": "D"},
        "validation": {"valid": True, "errors": [], "warnings": []},
        "config": {},
    }
    yield "test_fmt_parent"
    executor._data_handles.pop("test_fmt_parent", None)


def test_format_keeps_input_handle_alive(data_handle):
    executor = get_executor()
    result = transform_data_tool(data_handle=data_handle, action="format")
    assert result["success"], result
    new_handle = result["data_handle"]
    try:
        assert new_handle != data_handle
        assert data_handle in executor._data_handles, (
            "format released the caller's input handle"
        )
        assert new_handle in executor._data_handles
    finally:
        executor._data_handles.pop(new_handle, None)


def test_auto_format_on_load_still_releases_raw_handle():
    """The internal load path must not leak the raw pre-format handle."""
    executor = get_executor()
    before = set(executor._data_handles)
    res = executor.load_data_source(
        {
            "type": "pandas",
            "data": {
                "date": [f"2024-01-{d:02d}" for d in range(1, 11)],
                "value": list(range(10)),
            },
            "time_column": "date",
            "target_column": "value",
        }
    )
    assert res["success"], res
    created = set(executor._data_handles) - before
    try:
        # Exactly one new handle: the formatted one; the raw handle is gone
        assert created == {res["data_handle"]}
    finally:
        executor._data_handles.pop(res["data_handle"], None)
