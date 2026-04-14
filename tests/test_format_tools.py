"""
Tests for format_time_series_tool and executor.format_data_handle.

Covers the formatting tool in ``sktime_mcp.tools.format_tools`` and the
underlying ``Executor.format_data_handle`` method that it delegates to.

* ``format_time_series_tool`` – thin tool wrapper
* ``executor.format_data_handle`` – duplicate removal, frequency inference,
  gap filling, missing value imputation
"""

import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, "src")

from sktime_mcp.runtime.executor import get_executor
from sktime_mcp.tools.format_tools import format_time_series_tool


def _inject_data_handle(executor, y, X=None, metadata=None):
    """Insert a raw data handle into the executor, bypassing auto-format.

    Returns the data_handle string.
    """
    import uuid

    handle = f"data_{uuid.uuid4().hex[:8]}"
    executor._data_handles[handle] = {
        "y": y,
        "X": X,
        "metadata": metadata or {"rows": len(y), "source": "test"},
        "validation": {},
        "config": {},
    }
    return handle


def _cleanup_handles(executor, *handles):
    """Remove data handles from the executor."""
    for h in handles:
        executor._data_handles.pop(h, None)


class TestFormatTimeSeriesToolCore:
    """Core functionality tests for format_time_series_tool."""

    def test_format_clean_daily_data(self):
        """Clean daily data with no issues should still succeed."""
        executor = get_executor()
        dates = pd.date_range("2020-01-01", periods=30, freq="D")
        y = pd.Series(range(30), index=dates, dtype=float)
        handle = _inject_data_handle(executor, y)

        try:
            result = format_time_series_tool(handle)

            assert result["success"], f"Formatting failed: {result.get('error')}"
            assert "data_handle" in result
            assert "changes_made" in result
            assert "metadata" in result
            # New handle should differ from original
            assert result["data_handle"] != handle
            assert result["metadata"]["formatted"] is True
        finally:
            _cleanup_handles(executor, handle, result.get("data_handle", ""))

    def test_format_returns_new_handle(self):
        """Formatting should always create a new data handle, not modify in-place."""
        executor = get_executor()
        dates = pd.date_range("2020-01-01", periods=10, freq="D")
        y = pd.Series(range(10), index=dates, dtype=float)
        handle = _inject_data_handle(executor, y)

        try:
            result = format_time_series_tool(handle)

            assert result["success"]
            new_handle = result["data_handle"]
            # Both the old and new handle should exist
            assert handle in executor._data_handles
            assert new_handle in executor._data_handles
        finally:
            _cleanup_handles(executor, handle, result.get("data_handle", ""))

    def test_format_preserves_original_data(self):
        """The original handle's data should be unchanged after formatting."""
        executor = get_executor()
        dates = pd.date_range("2020-01-01", periods=10, freq="D")
        y = pd.Series(range(10), index=dates, dtype=float)
        y.iloc[3] = np.nan  # inject a missing value
        original_y = y.copy()
        handle = _inject_data_handle(executor, y)

        try:
            result = format_time_series_tool(handle)
            assert result["success"]

            # Original data should still have the NaN
            stored_original = executor._data_handles[handle]["y"]
            assert stored_original.isna().sum() == 1
            pd.testing.assert_series_equal(stored_original, original_y)
        finally:
            _cleanup_handles(executor, handle, result.get("data_handle", ""))

    def test_format_metadata_contains_key_fields(self):
        """Result metadata should include formatted flag, frequency, rows, dates."""
        executor = get_executor()
        dates = pd.date_range("2020-01-01", periods=20, freq="D")
        y = pd.Series(range(20), index=dates, dtype=float)
        handle = _inject_data_handle(executor, y)

        try:
            result = format_time_series_tool(handle)
            assert result["success"]

            meta = result["metadata"]
            assert meta["formatted"] is True
            assert "frequency" in meta
            assert "rows" in meta
            assert "start_date" in meta
            assert "end_date" in meta
        finally:
            _cleanup_handles(executor, handle, result.get("data_handle", ""))


class TestFormatDuplicateRemoval:
    """Tests for duplicate timestamp removal."""

    def test_removes_duplicate_timestamps(self):
        """Duplicate index entries should be removed (keep first)."""
        executor = get_executor()
        dates = pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-02", "2020-01-03"])
        y = pd.Series([10.0, 20.0, 25.0, 30.0], index=dates)
        handle = _inject_data_handle(executor, y)

        try:
            result = format_time_series_tool(handle, remove_duplicates=True)

            assert result["success"]
            assert result["changes_made"]["duplicates_removed"] > 0

            new_y = executor._data_handles[result["data_handle"]]["y"]
            assert not new_y.index.duplicated().any()
            # The first occurrence (20.0) should be kept, not the duplicate (25.0)
            assert new_y.loc["2020-01-02"] == 20.0
        finally:
            _cleanup_handles(executor, handle, result.get("data_handle", ""))

    def test_no_duplicates_reports_zero(self):
        """Data without duplicates should report duplicates_removed=0."""
        executor = get_executor()
        dates = pd.date_range("2020-01-01", periods=5, freq="D")
        y = pd.Series(range(5), index=dates, dtype=float)
        handle = _inject_data_handle(executor, y)

        try:
            result = format_time_series_tool(handle, remove_duplicates=True)

            assert result["success"]
            assert result["changes_made"]["duplicates_removed"] == 0
        finally:
            _cleanup_handles(executor, handle, result.get("data_handle", ""))

    def test_skip_duplicate_removal_when_disabled(self):
        """When remove_duplicates=False, duplicates should be preserved."""
        executor = get_executor()
        dates = pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-02", "2020-01-03"])
        y = pd.Series([10.0, 20.0, 25.0, 30.0], index=dates)
        handle = _inject_data_handle(executor, y)

        try:
            result = format_time_series_tool(
                handle, remove_duplicates=False, auto_infer_freq=False, fill_missing=False,
            )

            assert result["success"]
            assert result["changes_made"]["duplicates_removed"] == 0
        finally:
            _cleanup_handles(executor, handle, result.get("data_handle", ""))


class TestFormatFrequencyInference:
    """Tests for automatic frequency inference and gap filling."""

    def test_infers_daily_frequency(self):
        """Regular daily data should be inferred as 'D' frequency."""
        executor = get_executor()
        # Create daily data without explicit freq
        dates = pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04", "2020-01-05"])
        y = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], index=dates)
        handle = _inject_data_handle(executor, y)

        try:
            result = format_time_series_tool(handle, auto_infer_freq=True)

            assert result["success"]
            assert result["changes_made"]["frequency_set"] is True
        finally:
            _cleanup_handles(executor, handle, result.get("data_handle", ""))

    def test_fills_gaps_in_daily_data(self):
        """Missing dates in daily data should be filled via reindex."""
        executor = get_executor()
        # Day 3 is missing → gap of 1
        dates = pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-04", "2020-01-05"])
        y = pd.Series([1.0, 2.0, 4.0, 5.0], index=dates)
        handle = _inject_data_handle(executor, y)

        try:
            result = format_time_series_tool(handle, auto_infer_freq=True, fill_missing=True)

            assert result["success"]
            assert result["changes_made"]["gaps_filled"] > 0

            # Formatted data should have 5 rows (Jan 1-5 inclusive)
            new_y = executor._data_handles[result["data_handle"]]["y"]
            assert len(new_y) == 5
            # The gap-filled value should not be NaN after fill_missing
            assert not new_y.isna().any()
        finally:
            _cleanup_handles(executor, handle, result.get("data_handle", ""))

    def test_skip_freq_inference_when_disabled(self):
        """With auto_infer_freq=False, frequency should not be set."""
        executor = get_executor()
        dates = pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-04"])
        y = pd.Series([1.0, 2.0, 4.0], index=dates)
        handle = _inject_data_handle(executor, y)

        try:
            result = format_time_series_tool(
                handle, auto_infer_freq=False, fill_missing=False, remove_duplicates=False,
            )

            assert result["success"]
            assert result["changes_made"]["frequency_set"] is False
            assert result["changes_made"]["gaps_filled"] == 0
        finally:
            _cleanup_handles(executor, handle, result.get("data_handle", ""))

    def test_infers_hourly_frequency(self):
        """Hourly time series should be detected and frequency set."""
        executor = get_executor()
        dates = pd.date_range("2020-01-01", periods=24, freq="h")
        y = pd.Series(range(24), index=dates, dtype=float)
        # Remove the explicit freq to simulate raw data
        y.index = pd.DatetimeIndex(y.index.values)
        handle = _inject_data_handle(executor, y)

        try:
            result = format_time_series_tool(handle, auto_infer_freq=True)

            assert result["success"]
            assert result["changes_made"]["frequency_set"] is True
        finally:
            _cleanup_handles(executor, handle, result.get("data_handle", ""))


class TestFormatMissingValueFill:
    """Tests for missing value (NaN) filling."""

    def test_fills_nan_values(self):
        """NaN values should be filled using forward/backward fill."""
        executor = get_executor()
        dates = pd.date_range("2020-01-01", periods=10, freq="D")
        y = pd.Series(range(10), index=dates, dtype=float)
        y.iloc[3] = np.nan
        y.iloc[7] = np.nan
        handle = _inject_data_handle(executor, y)

        try:
            result = format_time_series_tool(handle, fill_missing=True, auto_infer_freq=False)

            assert result["success"]
            assert result["changes_made"]["missing_filled"] == 2

            new_y = executor._data_handles[result["data_handle"]]["y"]
            assert not new_y.isna().any()
        finally:
            _cleanup_handles(executor, handle, result.get("data_handle", ""))

    def test_no_nans_reports_zero(self):
        """Data without NaNs should report missing_filled=0."""
        executor = get_executor()
        dates = pd.date_range("2020-01-01", periods=10, freq="D")
        y = pd.Series(range(10), index=dates, dtype=float)
        handle = _inject_data_handle(executor, y)

        try:
            result = format_time_series_tool(handle, fill_missing=True, auto_infer_freq=False)

            assert result["success"]
            assert result["changes_made"]["missing_filled"] == 0
        finally:
            _cleanup_handles(executor, handle, result.get("data_handle", ""))

    def test_skip_nan_fill_when_disabled(self):
        """With fill_missing=False, NaN values should remain."""
        executor = get_executor()
        dates = pd.date_range("2020-01-01", periods=5, freq="D")
        y = pd.Series([1.0, np.nan, 3.0, np.nan, 5.0], index=dates)
        handle = _inject_data_handle(executor, y)

        try:
            result = format_time_series_tool(
                handle, fill_missing=False, auto_infer_freq=False, remove_duplicates=False,
            )

            assert result["success"]
            assert result["changes_made"]["missing_filled"] == 0

            new_y = executor._data_handles[result["data_handle"]]["y"]
            assert new_y.isna().sum() == 2
        finally:
            _cleanup_handles(executor, handle, result.get("data_handle", ""))


class TestFormatWithExogenousData:
    """Tests for formatting when exogenous (X) data is present."""

    def test_x_data_is_also_formatted(self):
        """Exogenous data should receive the same formatting as y."""
        executor = get_executor()
        dates = pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-02", "2020-01-03"])
        y = pd.Series([10.0, 20.0, 25.0, 30.0], index=dates)
        X = pd.DataFrame({"feat": [100.0, 200.0, 250.0, 300.0]}, index=dates)
        handle = _inject_data_handle(executor, y, X=X)

        try:
            result = format_time_series_tool(handle, remove_duplicates=True)

            assert result["success"]
            new_data = executor._data_handles[result["data_handle"]]
            new_X = new_data["X"]
            assert new_X is not None
            # X should also have duplicates removed
            assert not new_X.index.duplicated().any()
            # X and y should be aligned
            assert len(new_data["y"]) == len(new_X)
        finally:
            _cleanup_handles(executor, handle, result.get("data_handle", ""))

    def test_none_x_stays_none(self):
        """When X=None, the formatted data should also have X=None."""
        executor = get_executor()
        dates = pd.date_range("2020-01-01", periods=5, freq="D")
        y = pd.Series(range(5), index=dates, dtype=float)
        handle = _inject_data_handle(executor, y, X=None)

        try:
            result = format_time_series_tool(handle)

            assert result["success"]
            new_data = executor._data_handles[result["data_handle"]]
            assert new_data["X"] is None
        finally:
            _cleanup_handles(executor, handle, result.get("data_handle", ""))


class TestFormatInvalidInputs:
    """Tests for invalid inputs and error handling."""

    def test_unknown_data_handle(self):
        """Non-existent data handle should return success=False."""
        result = format_time_series_tool("data_nonexistent_xyz")

        assert not result["success"]
        assert "error" in result
        assert "not found" in result["error"].lower()

    def test_empty_string_handle(self):
        """Empty string handle should return success=False."""
        result = format_time_series_tool("")

        assert not result["success"]
        assert "error" in result

    def test_format_same_handle_twice(self):
        """Formatting the same handle twice should produce two new handles."""
        executor = get_executor()
        dates = pd.date_range("2020-01-01", periods=10, freq="D")
        y = pd.Series(range(10), index=dates, dtype=float)
        handle = _inject_data_handle(executor, y)

        new_handles = []
        try:
            result1 = format_time_series_tool(handle)
            result2 = format_time_series_tool(handle)

            assert result1["success"]
            assert result2["success"]
            # Each call should produce a distinct handle
            assert result1["data_handle"] != result2["data_handle"]
            new_handles = [result1["data_handle"], result2["data_handle"]]
        finally:
            _cleanup_handles(executor, handle, *new_handles)


class TestFormatAllOptionsDisabled:
    """Boundary test: all formatting options disabled."""

    def test_all_disabled_still_succeeds(self):
        """Disabling all options should still succeed (sort only)."""
        executor = get_executor()
        dates = pd.date_range("2020-01-01", periods=10, freq="D")
        y = pd.Series(range(10), index=dates, dtype=float)
        y.iloc[2] = np.nan
        handle = _inject_data_handle(executor, y)

        try:
            result = format_time_series_tool(
                handle,
                auto_infer_freq=False,
                fill_missing=False,
                remove_duplicates=False,
            )

            assert result["success"]
            changes = result["changes_made"]
            assert changes["frequency_set"] is False
            assert changes["duplicates_removed"] == 0
            assert changes["missing_filled"] == 0
            assert changes["gaps_filled"] == 0

            # NaN should still be present
            new_y = executor._data_handles[result["data_handle"]]["y"]
            assert new_y.isna().sum() == 1
        finally:
            _cleanup_handles(executor, handle, result.get("data_handle", ""))


class TestFormatDataHandleDirectly:
    """Tests that call executor.format_data_handle directly."""

    def test_unsorted_data_gets_sorted(self):
        """Data with reversed index should be sorted after formatting."""
        executor = get_executor()
        dates = pd.to_datetime(["2020-01-05", "2020-01-03", "2020-01-01", "2020-01-04", "2020-01-02"])
        y = pd.Series([5.0, 3.0, 1.0, 4.0, 2.0], index=dates)
        handle = _inject_data_handle(executor, y)

        try:
            result = executor.format_data_handle(
                handle, auto_infer_freq=False, fill_missing=False, remove_duplicates=False,
            )

            assert result["success"]
            new_y = executor._data_handles[result["data_handle"]]["y"]
            # Index should be monotonically increasing
            assert new_y.index.is_monotonic_increasing
        finally:
            _cleanup_handles(executor, handle, result.get("data_handle", ""))

    def test_tracks_original_handle(self):
        """Formatted data should reference original_handle for traceability."""
        executor = get_executor()
        dates = pd.date_range("2020-01-01", periods=5, freq="D")
        y = pd.Series(range(5), index=dates, dtype=float)
        handle = _inject_data_handle(executor, y)

        try:
            result = executor.format_data_handle(handle)

            assert result["success"]
            stored = executor._data_handles[result["data_handle"]]
            assert stored["original_handle"] == handle
        finally:
            _cleanup_handles(executor, handle, result.get("data_handle", ""))

    def test_combined_duplicates_gaps_nans(self):
        """Data with all three problems should report all fixes."""
        executor = get_executor()
        # Build data with: duplicate, gap, NaN
        dates = pd.to_datetime([
            "2020-01-01",
            "2020-01-02",
            "2020-01-02",  # duplicate
            "2020-01-04",  # gap (Jan 3 missing)
            "2020-01-05",
        ])
        y = pd.Series([1.0, 2.0, 2.5, np.nan, 5.0], index=dates)
        handle = _inject_data_handle(executor, y)

        try:
            result = executor.format_data_handle(
                handle, auto_infer_freq=True, fill_missing=True, remove_duplicates=True,
            )

            assert result["success"]
            changes = result["changes_made"]
            assert changes["duplicates_removed"] > 0
            assert changes["gaps_filled"] > 0
            # After reindex + ffill/bfill, no NaNs should remain
            new_y = executor._data_handles[result["data_handle"]]["y"]
            assert not new_y.isna().any()
        finally:
            _cleanup_handles(executor, handle, result.get("data_handle", ""))


class TestFormatIntegrationWithLoadSource:
    """Integration: load data via executor, then format explicitly."""

    def test_format_after_load_data_source(self):
        """Data loaded via load_data_source should be formattable."""
        executor = get_executor()
        # Disable auto-format so we get raw data
        old_setting = executor._auto_format_enabled
        executor._auto_format_enabled = False

        try:
            config = {
                "type": "pandas",
                "data": {
                    "date": pd.date_range("2020-01-01", periods=20, freq="D"),
                    "value": list(range(20)),
                },
                "time_column": "date",
                "target_column": "value",
            }
            load_result = executor.load_data_source(config)
            assert load_result["success"], f"Load failed: {load_result}"
            data_handle = load_result["data_handle"]

            fmt_result = format_time_series_tool(data_handle)
            assert fmt_result["success"]
            assert fmt_result["metadata"]["formatted"] is True
        finally:
            executor._auto_format_enabled = old_setting
            _cleanup_handles(
                executor,
                load_result.get("data_handle", ""),
                fmt_result.get("data_handle", ""),
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
