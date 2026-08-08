"""Not-found error responses must not dump the entire handle store.

available_handles is capped at the 5 most recent handles, with the total
reported separately — a full dump both enumerates other sessions' handles
in a shared store and floods the caller's context.
"""

import pandas as pd
import pytest

from sktime_mcp.runtime.executor import get_executor
from sktime_mcp.tools.inspect_data import inspect_data_tool
from sktime_mcp.tools.plotting import plot_series_tool


@pytest.fixture(autouse=True)
def _clean_handles():
    executor = get_executor()
    executor._data_handles.clear()
    yield
    executor._data_handles.clear()


def _register_handles(n: int) -> list[str]:
    executor = get_executor()
    series = pd.Series(range(10), index=pd.date_range("2020-01", periods=10, freq="MS"))
    ids = [f"data_test_{i:02d}" for i in range(n)]
    for handle_id in ids:
        executor._data_handles[handle_id] = {"y": series}
    return ids


def test_summary_capped_at_five_most_recent():
    ids = _register_handles(8)
    summary = get_executor().summarize_available_handles()
    assert summary["available_handles"] == ids[-5:]
    assert summary["n_available_handles"] == 8


def test_summary_below_cap_lists_all():
    ids = _register_handles(3)
    summary = get_executor().summarize_available_handles()
    assert summary["available_handles"] == ids
    assert summary["n_available_handles"] == 3


def test_inspect_data_not_found_is_capped():
    _register_handles(8)
    result = inspect_data_tool(data_handle="nonexistent")
    assert result["success"] is False
    assert len(result["available_handles"]) == 5
    assert result["n_available_handles"] == 8


def test_plot_series_not_found_is_capped():
    _register_handles(8)
    result = plot_series_tool(data_handles=["nonexistent"])
    assert result["success"] is False
    assert len(result["available_handles"]) == 5
    assert result["n_available_handles"] == 8
