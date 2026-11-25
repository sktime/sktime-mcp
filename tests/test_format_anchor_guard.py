"""Explicit frequency preparation preserves observations and never guesses an anchor."""

import pandas as pd
import pytest

from sktime_mcp.runtime.executor import Executor


@pytest.mark.parametrize(
    "freq,start,count",
    [
        ("W-MON", "2024-01-01", 12),
        ("30min", "2024-01-01", 100),
        ("QS", "2022-01-01", 12),
        ("D", "2023-01-01", 10),
        ("MS", "2023-01-01", 10),
    ],
)
def test_explicit_aligned_frequency_keeps_original_values(freq, start, count):
    ex = Executor()
    original = pd.Series(
        range(count - 1),
        index=pd.date_range(start, periods=count, freq=freq).delete(4),
        dtype=float,
    )
    load = ex.load_data({"type": "memory", "data": original})
    assert load["success"]
    pd.testing.assert_series_equal(ex._data_handles[load["data_handle"]]["y"], original)
    prepared = ex.prepare_data(
        load["data_handle"], "format", frequency=freq, regularize=True, missing="ffill"
    )
    assert prepared["success"], prepared
    actual = ex._data_handles[prepared["data_handle"]]["y"]
    assert len(actual) == count
    pd.testing.assert_series_equal(actual.loc[original.index], original, check_freq=False)
    assert not actual.isna().any()
    assert prepared["changes_applied"]["y"]["gaps_filled"] == 1
    assert prepared["changes_applied"]["y"]["missing_filled"] == 1
    pd.testing.assert_series_equal(ex._data_handles[load["data_handle"]]["y"], original)


@pytest.mark.parametrize(
    "dates,freq",
    [
        (["2024-01-15", "2024-02-15", "2024-04-15"], "MS"),
        (["2024-01-01", "2024-01-08", "2024-01-22"], "W-SUN"),
        (["2024-01-01 00:00", "2024-01-01 00:30", "2024-01-01 01:01"], "30min"),
    ],
)
def test_mismatched_anchor_fails_without_losing_data(dates, freq):
    ex = Executor()
    original = pd.Series([1.0, 2.0, 3.0], index=pd.to_datetime(dates))
    load = ex.load_data({"type": "memory", "data": original})
    before = set(ex._data_handles)
    result = ex.prepare_data(load["data_handle"], "format", frequency=freq, regularize=True)
    assert not result["success"]
    assert "does not align" in result["error"]
    assert set(ex._data_handles) == before
    pd.testing.assert_series_equal(ex._data_handles[load["data_handle"]]["y"], original)


def test_no_frequency_is_guessed_for_midmonth_data():
    ex = Executor()
    dates = pd.date_range("2024-01-15", periods=13, freq=pd.DateOffset(months=1)).delete(5)
    original = pd.Series(range(12), index=dates)
    load = ex.load_data({"type": "memory", "data": original})
    result = ex.prepare_data(load["data_handle"], "format", sort_index=True)
    assert result["success"]
    assert result["changes_applied"]["y"]["gaps_filled"] == 0
    pd.testing.assert_series_equal(ex._data_handles[result["data_handle"]]["y"], original)


def test_repair_counts_match_actual_target_and_feature_changes():
    ex = Executor()
    original = pd.DataFrame(
        {"y": [1.0, None, 3.0, 9.0], "X": [None, 2.0, None, 9.0]},
        index=pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-04", "2024-01-02"]),
    )
    loaded = ex.load_data({"type": "memory", "data": original, "target_column": "y"})
    result = ex.prepare_data(
        loaded["data_handle"],
        "format",
        sort_index=True,
        duplicates="first",
        frequency="D",
        regularize=True,
        missing="ffill",
    )
    assert result["success"], result
    changes = result["changes_applied"]
    assert changes["y"]["duplicates_removed"] == 1
    assert changes["y"]["gaps_filled"] == 1
    assert changes["y"]["missing_filled"] == 2
    assert changes["X"]["missing_filled"] == 2
    assert result["inspection"]["n_missing"] == 0
    assert result["inspection"]["exogenous"]["n_missing"] == 1
    for part in changes.values():
        for name in ["duplicates_removed", "gaps_filled", "missing_filled"]:
            assert type(part[name]) is int and part[name] >= 0
