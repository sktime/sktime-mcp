"""Handle-loaded monthly data must support seasonal forecasters (#531).

load_data_source produced a DatetimeIndex with freq "MS" (MonthBegin), while
seasonal forecasters coerce to PeriodIndex internally and raise
"<MonthBegin> is not supported as period frequency" at predict time — fit
succeeded, predict failed. Handles are now normalised to PeriodIndex at load,
matching demo datasets.
"""

import pandas as pd
import pytest

from sktime_mcp.runtime.executor import _to_period_index_if_possible, get_executor
from sktime_mcp.tools.fit_predict import fit_tool, predict_tool, update_tool
from sktime_mcp.tools.instantiate import instantiate_tool


def _load_monthly(executor, periods=36):
    data = {
        "date": [f"20{20 + i // 12}-{i % 12 + 1:02d}-01" for i in range(periods)],
        "value": [100.0 + i + 10 * (i % 12) for i in range(periods)],
    }
    res = executor.load_data_source(
        {"type": "pandas", "data": data, "time_column": "date", "target_column": "value"}
    )
    assert res["success"], res
    return res["data_handle"]


class TestHelper:
    def test_datetime_ms_becomes_period(self):
        idx = pd.date_range("2020-01-01", periods=12, freq="MS")
        s = pd.Series(range(12), index=idx)
        out = _to_period_index_if_possible(s)
        assert isinstance(out.index, pd.PeriodIndex)
        assert out.index.freqstr == "M"

    def test_freqless_but_regular_is_inferred(self):
        idx = pd.DatetimeIndex(pd.date_range("2020-01-01", periods=12, freq="MS").values)
        assert idx.freq is None
        out = _to_period_index_if_possible(pd.Series(range(12), index=idx))
        assert isinstance(out.index, pd.PeriodIndex)

    def test_irregular_index_untouched(self):
        idx = pd.to_datetime(["2020-01-01", "2020-01-05", "2020-03-02"])
        s = pd.Series([1, 2, 3], index=idx)
        out = _to_period_index_if_possible(s)
        assert isinstance(out.index, pd.DatetimeIndex)

    def test_period_index_noop(self):
        s = pd.Series(range(6), index=pd.period_range("2020-01", periods=6, freq="M"))
        assert _to_period_index_if_possible(s) is not None
        assert isinstance(_to_period_index_if_possible(s).index, pd.PeriodIndex)


class TestSeasonalPredictOnHandle:
    def test_load_gives_period_index(self):
        executor = get_executor()
        dh = _load_monthly(executor)
        try:
            assert isinstance(executor._data_handles[dh]["y"].index, pd.PeriodIndex)
        finally:
            executor._data_handles.pop(dh, None)

    def test_seasonal_fit_then_predict(self):
        """The exact reported repro: fit sp=12 on a handle, then predict."""
        executor = get_executor()
        dh = _load_monthly(executor)
        inst = instantiate_tool(spec="NaiveForecaster(strategy='last', sp=12)")
        handle = inst["handle"]
        try:
            fit_res = fit_tool(estimator_handle=handle, y_handle=dh)
            assert fit_res["success"], fit_res
            pred = predict_tool(estimator_handle=handle, horizon=6)
            assert pred["success"], pred
            assert len(pred["predictions"]) == 6
        finally:
            executor._handle_manager.release_handle(handle)
            executor._data_handles.pop(dh, None)

    def test_seasonal_evaluate_on_handle(self):
        from sktime_mcp.tools.evaluate import evaluate_tool

        executor = get_executor()
        dh = _load_monthly(executor)
        inst = instantiate_tool(spec="NaiveForecaster(sp=12)")
        handle = inst["handle"]
        try:
            res = evaluate_tool(estimator_handle=handle, y=dh, cv_folds=3)
            assert res["success"], res
            for v in res["metrics"].values():
                assert v == v  # not NaN
        finally:
            executor._handle_manager.release_handle(handle)
            executor._data_handles.pop(dh, None)
