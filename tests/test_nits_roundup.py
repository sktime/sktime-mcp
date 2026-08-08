"""Correctness/quality nits roundup (#541)."""

import contextlib

import pandas as pd
import pytest

from sktime_mcp.runtime.executor import get_executor
from sktime_mcp.runtime.handles import get_handle_manager
from sktime_mcp.tools.fit_predict import fit_tool, predict_tool
from sktime_mcp.tools.inspect_data import inspect_data_tool
from sktime_mcp.tools.instantiate import instantiate_tool, release_handle_tool
from sktime_mcp.tools.plotting import plot_series_tool


def _release(h):
    with contextlib.suppress(KeyError):
        get_handle_manager().release_handle(h)


def test_release_handle_failure_has_error_key():
    # NB-02: failure must carry an "error" key like other tools
    res = release_handle_tool("est_never_existed")
    assert res["success"] is False
    assert "error" in res
    assert "not found" in res["error"].lower()


def test_classifier_predict_omits_horizon():
    # N-01: horizon is meaningless for classifiers
    h = instantiate_tool(spec="KNeighborsTimeSeriesClassifier()")["handle"]
    try:
        fit_tool(estimator_handle=h, X_dataset="arrow_head", y_dataset="arrow_head")
        res = predict_tool(estimator_handle=h, X_dataset="arrow_head")
        assert res["success"], res
        assert "horizon" not in res
    finally:
        _release(h)


def test_forecaster_predict_keeps_horizon():
    h = instantiate_tool(spec="NaiveForecaster(sp=12)")["handle"]
    try:
        fit_tool(estimator_handle=h, y_dataset="airline")
        res = predict_tool(estimator_handle=h, horizon=6)
        assert res["horizon"] == 6
    finally:
        _release(h)


def test_format_reports_sorting():
    # NB-08: sorting is surfaced in changes
    ex = get_executor()
    ex._data_handles["nit_unsorted"] = {
        "y": pd.Series(
            [3.0, 1.0, 2.0],
            index=pd.to_datetime(["2024-03-01", "2024-01-01", "2024-02-01"]),
        ),
        "X": None,
        "metadata": {"frequency": None},
        "validation": {},
        "config": {},
    }
    try:
        res = ex.format_data_handle("nit_unsorted", release_original=False)
        assert res["success"]
        assert res["changes_made"].get("sorted") is True
    finally:
        for h in list(ex._data_handles):
            if h == "nit_unsorted" or h == res.get("data_handle"):
                ex._data_handles.pop(h, None)


def test_inspect_includes_exog_dtypes():
    # N-14: dtypes must include exog columns that `columns` lists
    ex = get_executor()
    idx = pd.period_range("2020-01", periods=12, freq="M")
    ex._data_handles["nit_exog"] = {
        "y": pd.Series(range(12), index=idx, name="target", dtype=float),
        "X": pd.DataFrame({"temp": range(12)}, index=idx),
        "metadata": {},
        "validation": {},
        "config": {},
    }
    try:
        res = inspect_data_tool(data_handle="nit_exog")
        assert res["success"], res
        assert "X:temp" in res["dtypes"]
        assert "X:temp" in res["columns"]
    finally:
        ex._data_handles.pop("nit_exog", None)


def test_plot_rejects_nonpositive_dpi():
    # N-20: dpi=0 must be rejected, not silently defaulted
    ex = get_executor()
    idx = pd.period_range("2020-01", periods=12, freq="M")
    ex._data_handles["nit_plot"] = {"y": pd.Series(range(12), index=idx)}
    try:
        res = plot_series_tool(data_handles=["nit_plot"], dpi=0)
        assert not res["success"]
        assert "dpi" in res["error"].lower()
    finally:
        ex._data_handles.pop("nit_plot", None)
