"""Tests for calendar-safe formatting (non-datetime indices must not be corrupted)."""

import sys

import pandas as pd
import pytest

sys.path.insert(0, "src")


@pytest.mark.parametrize(
    "index_factory,expect_skip",
    [
        (lambda: pd.RangeIndex(3), True),
        (lambda: pd.Index([10, 20, 30], dtype="int64"), True),
    ],
)
def test_calendar_gap_fill_skipped_for_non_datetime_index(index_factory, expect_skip):
    from sktime_mcp.runtime.executor import _apply_calendar_frequency_and_gap_fill

    y = pd.Series([1.0, 2.0, 3.0], index=index_factory())
    changes_made = {
        "frequency_set": False,
        "duplicates_removed": 0,
        "missing_filled": 0,
        "gaps_filled": 0,
    }

    y_out, X_out = _apply_calendar_frequency_and_gap_fill(y, None, changes_made)

    assert X_out is None
    assert len(y_out) == 3
    assert list(y_out.values) == pytest.approx([1.0, 2.0, 3.0])
    if expect_skip:
        assert changes_made.get("calendar_gap_fill_skipped") is True
        assert "Unsupported index type" in changes_made.get("calendar_gap_fill_reason", "")


def test_datetime_index_gap_fill_still_runs():
    from sktime_mcp.runtime.executor import _apply_calendar_frequency_and_gap_fill

    idx = pd.to_datetime(["2020-01-01", "2020-01-03"])
    y = pd.Series([1.0, 3.0], index=idx)
    changes_made = {
        "frequency_set": False,
        "duplicates_removed": 0,
        "missing_filled": 0,
        "gaps_filled": 0,
    }

    y_out, X_out = _apply_calendar_frequency_and_gap_fill(y, None, changes_made)

    assert X_out is None
    assert isinstance(y_out.index, pd.DatetimeIndex)
    assert changes_made.get("calendar_gap_fill_skipped") is not True
    assert len(y_out) >= 2


def test_format_data_handle_range_index_via_executor():
    """format_data_handle must not destroy RangeIndex series (no __init__ → no registry)."""
    from sktime_mcp.runtime.executor import Executor

    ex = Executor.__new__(Executor)
    ex._data_handles = {}

    h = "data_testfmt1"
    ex._data_handles[h] = {
        "y": pd.Series([10.0, 20.0, 30.0]),
        "X": None,
        "metadata": {"source": "test"},
        "validation": {},
        "config": {},
    }

    result = Executor.format_data_handle(ex, h, auto_infer_freq=True)
    assert result["success"]
    new_h = result["data_handle"]
    y_new = ex._data_handles[new_h]["y"]
    assert len(y_new) == 3
    assert list(y_new.values) == pytest.approx([10.0, 20.0, 30.0])
    assert result["changes_made"].get("calendar_gap_fill_skipped") is True
