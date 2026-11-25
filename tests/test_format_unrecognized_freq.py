"""Irregular timestamps survive loading and preparation without guessed resampling."""

import pandas as pd
import pytest

from sktime_mcp.runtime.executor import Executor

DATES = [
    "2023-01-02",
    "2023-01-16",
    "2023-01-30",
    "2023-02-13",
    "2023-02-28",
    "2023-03-13",
    "2023-03-27",
    "2023-04-10",
]


@pytest.mark.parametrize("prepare", [False, True])
def test_irregular_biweekly_not_expanded_to_daily(prepare):
    ex = Executor()
    original = pd.Series([10.0, 12, 11, 14, 13, 15, 12, 16], index=pd.to_datetime(DATES))
    result = ex.load_data({"type": "memory", "data": original})
    if prepare:
        result = ex.prepare_data(result["data_handle"], "format", sort_index=True)
    assert result["success"]
    pd.testing.assert_series_equal(ex._data_handles[result["data_handle"]]["y"], original)


@pytest.mark.parametrize("frequency", ["15min", "D"])
def test_gaps_filled_only_on_explicit_request(frequency):
    ex = Executor()
    index = pd.date_range("2023-01-01", periods=20, freq=frequency).delete(7)
    original = pd.Series(range(19), index=index, dtype=float)
    load = ex.load_data({"type": "memory", "data": original})
    assert load["metadata"]["rows"] == 19
    result = ex.prepare_data(load["data_handle"], "format", frequency=frequency, regularize=True)
    assert result["success"]
    actual = ex._data_handles[result["data_handle"]]["y"]
    assert len(actual) == 20
    assert actual.isna().sum() == 1
    pd.testing.assert_series_equal(actual.loc[index], original, check_freq=False)


def test_irregular_data_with_features_can_still_fit():
    ex = Executor()
    data = pd.DataFrame(
        {"value": [10.0, 12, 11, 14, 13, 15, 12, 16], "temp": range(8)}, index=pd.to_datetime(DATES)
    )
    load = ex.load_data({"type": "memory", "data": data, "target_column": "value"})
    model = ex.instantiate(spec="NaiveForecaster(strategy='last')")
    try:
        assert load["success"]
        record = ex._data_handles[load["data_handle"]]
        assert len(record["y"]) == len(record["X"]) == 8
        assert ex.fit(model["handle"], y=record["y"], X=record["X"])["success"]
        pd.testing.assert_index_equal(record["y"].index, data.index)
    finally:
        ex._handle_manager.release_handle(model["handle"])
