"""Unrecognized intervals must not be reindexed to daily (#313)."""

import pandas as pd

from sktime_mcp.runtime.executor import get_executor
from sktime_mcp.tools.fit_predict import fit_tool
from sktime_mcp.tools.inspect_data import inspect_data_tool
from sktime_mcp.tools.instantiate import instantiate_tool
from sktime_mcp.tools.transform_data import transform_data_tool

_BIWEEKLY = [
    "2023-01-02",
    "2023-01-16",
    "2023-01-30",
    "2023-02-13",
    "2023-02-28",
    "2023-03-13",
    "2023-03-27",
    "2023-04-10",
]


def _register(ex, handle, index, values=None):
    y = pd.Series(
        values if values is not None else range(len(index)),
        index=pd.DatetimeIndex(index),
        dtype=float,
        name="value",
    )
    ex._data_handles[handle] = {
        "y": y,
        "X": None,
        "metadata": {"rows": len(y), "frequency": None},
        "validation": {},
        "config": {},
    }
    return handle


def _pop(ex, *handles):
    for h in handles:
        if h:
            ex._data_handles.pop(h, None)


def test_irregular_biweekly_not_expanded_to_daily():
    ex = get_executor()
    src = _register(ex, "fmt_biweekly", _BIWEEKLY, [10, 12, 11, 14, 13, 15, 12, 16])
    res = None
    try:
        res = ex.format_data_handle(src, release_original=False)
        assert res["success"], res
        assert res["metadata"]["rows"] == 8
        assert not res["changes_made"].get("frequency_set")
        assert "frequency_warning" in res["changes_made"]
        assert "14 days" in res["changes_made"]["frequency_warning"]
        y = ex._data_handles[res["data_handle"]]["y"]
        assert len(y) == 8
        assert list(y.index) == list(pd.to_datetime(_BIWEEKLY))
    finally:
        _pop(ex, src, res.get("data_handle") if res else None)


def test_gapped_15min_not_collapsed_to_one_day():
    ex = get_executor()
    idx = pd.date_range("2023-01-01 00:00", periods=20, freq="15min").delete(7)
    src = _register(ex, "fmt_15min", idx)
    res = None
    try:
        res = ex.format_data_handle(src, release_original=False)
        assert res["success"], res
        assert res["metadata"]["rows"] == 19
        assert not res["changes_made"].get("frequency_set")
        assert "frequency_warning" in res["changes_made"]
        assert len(ex._data_handles[res["data_handle"]]["y"]) == 19
    finally:
        _pop(ex, src, res.get("data_handle") if res else None)


def test_daily_gap_still_filled():
    ex = get_executor()
    idx = pd.date_range("2023-01-01", periods=10, freq="D").delete(4)
    src = _register(ex, "fmt_daily_gap", idx)
    res = None
    try:
        res = ex.format_data_handle(src, release_original=False)
        assert res["success"], res
        assert res["changes_made"]["frequency_set"]
        assert res["changes_made"]["frequency"] == "D"
        assert res["metadata"]["rows"] == 10
        assert res["changes_made"]["gaps_filled"] == 1
    finally:
        _pop(ex, src, res.get("data_handle") if res else None)


def test_transform_data_surfaces_frequency_warning():
    ex = get_executor()
    src = _register(ex, "fmt_td_biweekly", _BIWEEKLY)
    res = None
    try:
        res = transform_data_tool(data_handle=src, action="format")
        assert res["success"], res
        assert res["metadata"]["rows"] == 8
        applied = " ".join(res["changes_applied"])
        assert "Reindexing skipped" in applied
        assert "set frequency to 'D'" not in applied
        assert "already clean" not in applied
    finally:
        _pop(ex, src, res.get("data_handle") if res else None)


def test_irregular_biweekly_load_inspect_fit_keeps_real_rows():
    ex = get_executor()
    load = ex.load_data_source(
        {
            "type": "pandas",
            "data": {
                "date": _BIWEEKLY,
                "value": [10.0, 12, 11, 14, 13, 15, 12, 16],
                "temp": [1.0, 2, 3, 4, 5, 6, 7, 8],
            },
            "time_column": "date",
            "target_column": "value",
        }
    )
    inst = instantiate_tool(spec="NaiveForecaster(strategy='last')")
    handle = inst["handle"]
    try:
        assert load["success"], load
        assert not load["changes_made"]["frequency_set"]
        assert "frequency_warning" in load["changes_made"]
        dh = load["data_handle"]
        y = ex._data_handles[dh]["y"]
        assert len(y) == 8
        assert isinstance(y.index, pd.DatetimeIndex)
        X = ex._data_handles[dh]["X"]
        assert X is not None and len(X) == 8
        inspected = inspect_data_tool(dh)
        assert inspected["success"], inspected
        assert inspected["shape"][0] == 8
        fit = fit_tool(estimator_handle=handle, y_handle=dh)
        assert fit["success"], fit
    finally:
        _pop(ex, load.get("data_handle"))
        ex._handle_manager.release_handle(handle)
