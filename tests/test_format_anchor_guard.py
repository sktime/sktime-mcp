"""Auto-format must never reindex onto a mis-anchored frequency (#556 item 1, F-01/F-63).

``format_data_handle`` used to guess a frequency from the modal interval and
``reindex`` onto ``pd.date_range(min, max, freq)``. When the guessed anchor did
not match the timestamps (monthly data on the 15th -> "MS", weekly Mondays ->
"W"=W-SUN) every value was dropped and replaced by NaN, while the load response
still reported ``missing_values: 0``. Now an anchored frequency is tried and
the index is only reindexed when it is a subset of the generated range;
otherwise the series is left untouched with a ``frequency_warning``.
"""

import pandas as pd

from sktime_mcp.runtime.executor import get_executor
from sktime_mcp.tools.inspect_data import inspect_data_tool
from sktime_mcp.tools.transform_data import transform_data_tool


def _dates(start, periods, freq, drop):
    idx = pd.date_range(start, periods=periods, freq=freq).delete(drop)
    return [str(ts) for ts in idx]


def _load(ex, dates, values=None, name="sales"):
    values = values if values is not None else [float(i + 1) for i in range(len(dates))]
    res = ex.load_data_source(
        {
            "type": "pandas",
            "data": {"date": dates, name: values},
            "time_column": "date",
            "target_column": name,
        }
    )
    assert res["success"], res
    return res


def _pop(ex, *handles):
    for h in handles:
        if h:
            ex._data_handles.pop(h, None)


def _register(ex, handle, dates, values=None):
    y = pd.Series(
        values if values is not None else [float(i + 1) for i in range(len(dates))],
        index=pd.DatetimeIndex(dates),
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


def _assert_values_preserved(y, dates, values):
    """Every original (timestamp, value) pair survives formatting, and nothing is NaN."""
    assert not y.isna().any(), y
    stamps = pd.DatetimeIndex(dates)
    keys = stamps.to_period(y.index.freq) if isinstance(y.index, pd.PeriodIndex) else stamps
    for key, value in zip(keys, values, strict=True):
        assert key in y.index, (key, y.index)
        assert y.loc[key] == value, (key, y.loc[key], value)


class TestAnchoredFrequencies:
    def test_mid_month_monthly_gap_keeps_values(self):
        ex = get_executor()
        dates = _dates("2024-01-15", 13, pd.DateOffset(months=1), 5)
        values = [float(v) for v in range(100, 100 + len(dates))]
        res = _load(ex, dates, values)
        try:
            y = ex._data_handles[res["data_handle"]]["y"]
            _assert_values_preserved(y, dates, values)
            assert len(y) == 13
            changes = res["changes_made"]
            assert changes["gaps_filled"] == 1
            assert "frequency_warning" in changes
            assert res["metadata"]["missing_values"] == {"sales": 0}
            inspected = inspect_data_tool(res["data_handle"])
            assert inspected["success"], inspected
            assert inspected["n_missing"] == 0
            assert inspected["shape"][0] == 13
        finally:
            _pop(ex, res["data_handle"])

    def test_weekly_monday_gap_keeps_values(self):
        ex = get_executor()
        dates = _dates("2024-01-01", 12, "W-MON", 5)
        values = [float(v) for v in range(len(dates))]
        res = _load(ex, dates, values)
        try:
            y = ex._data_handles[res["data_handle"]]["y"]
            _assert_values_preserved(y, dates, values)
            assert len(y) == 12
            assert res["changes_made"]["frequency"] == "W-MON"
            assert res["changes_made"]["gaps_filled"] == 1
        finally:
            _pop(ex, res["data_handle"])

    def test_30min_gap_keeps_values(self):
        ex = get_executor()
        dates = _dates("2024-01-01 00:00", 100, "30min", 40)
        values = [float(v) for v in range(len(dates))]
        res = _load(ex, dates, values)
        try:
            y = ex._data_handles[res["data_handle"]]["y"]
            _assert_values_preserved(y, dates, values)
            assert len(y) == 100
            assert res["changes_made"]["frequency"] == "30min"
            assert res["changes_made"]["gaps_filled"] == 1
        finally:
            _pop(ex, res["data_handle"])

    def test_quarterly_gap_keeps_values(self):
        ex = get_executor()
        dates = _dates("2022-01-01", 12, "QS", 4)
        values = [float(v) for v in range(len(dates))]
        res = _load(ex, dates, values)
        try:
            y = ex._data_handles[res["data_handle"]]["y"]
            _assert_values_preserved(y, dates, values)
            assert len(y) == 12
            assert res["changes_made"]["frequency_set"]
            assert res["changes_made"]["gaps_filled"] == 1
        finally:
            _pop(ex, res["data_handle"])

    def test_daily_gap_still_filled(self):
        ex = get_executor()
        dates = _dates("2023-01-01", 10, "D", 4)
        values = [float(v) for v in range(len(dates))]
        res = _load(ex, dates, values)
        try:
            y = ex._data_handles[res["data_handle"]]["y"]
            _assert_values_preserved(y, dates, values)
            assert len(y) == 10
            assert res["changes_made"]["frequency"] == "D"
            assert res["changes_made"]["gaps_filled"] == 1
        finally:
            _pop(ex, res["data_handle"])


class TestMismatchedAnchorLeavesSeriesUnchanged:
    # Mostly the 15th, one reading on the 20th: no monthly anchor fits.
    DATES = [
        "2024-01-15",
        "2024-02-15",
        "2024-03-15",
        "2024-04-20",
        "2024-05-15",
        "2024-06-15",
        "2024-08-15",
    ]

    def test_load_keeps_series_and_warns(self):
        ex = get_executor()
        values = [float(v) for v in range(len(self.DATES))]
        res = _load(ex, self.DATES, values)
        try:
            y = ex._data_handles[res["data_handle"]]["y"]
            assert isinstance(y.index, pd.DatetimeIndex)
            assert list(y.index) == list(pd.to_datetime(self.DATES))
            assert list(y) == values
            changes = res["changes_made"]
            assert not changes["frequency_set"]
            assert changes["gaps_filled"] == 0
            assert changes["missing_filled"] == 0
            assert "does not align" in changes["frequency_warning"]
            assert "left unchanged" in changes["frequency_warning"]
            assert res["metadata"]["rows"] == len(self.DATES)
        finally:
            _pop(ex, res["data_handle"])

    def test_transform_format_applies_same_guard(self):
        ex = get_executor()
        values = [float(v) for v in range(len(self.DATES))]
        src = _register(ex, "fmt_anchor_mismatch", self.DATES, values)
        res = None
        try:
            res = transform_data_tool(data_handle=src, action="format")
            assert res["success"], res
            y = ex._data_handles[res["data_handle"]]["y"]
            assert list(y) == values
            assert list(y.index) == list(pd.to_datetime(self.DATES))
            applied = " ".join(res["changes_applied"])
            assert "does not align" in applied
            assert "set frequency" not in applied
        finally:
            _pop(ex, src, res.get("data_handle") if res else None)


class TestChangesMadeCounts:
    def test_counts_are_plain_ints_and_non_negative(self):
        ex = get_executor()
        dates = _dates("2023-01-01", 8, "D", 3) + ["2023-01-02 00:00:00"]
        values = [1.0, 2.0, None, 4.0, 5.0, 6.0, 7.0, 2.0]
        res = _load(ex, dates, values)
        try:
            changes = res["changes_made"]
            for key in ("duplicates_removed", "missing_filled", "gaps_filled"):
                assert type(changes[key]) is int, (key, type(changes[key]))
                assert changes[key] >= 0, (key, changes[key])
            assert changes["duplicates_removed"] == 1
            assert changes["gaps_filled"] == 1
            assert changes["missing_filled"] == 2
        finally:
            _pop(ex, res["data_handle"])

    def test_missing_values_describes_stored_data(self):
        ex = get_executor()
        dates = _dates("2023-01-01", 6, "D", [])
        values = [1.0, None, 3.0, 4.0, None, 6.0]
        res = _load(ex, dates, values)
        try:
            y = ex._data_handles[res["data_handle"]]["y"]
            assert int(y.isna().sum()) == 0
            assert res["metadata"]["missing_values"] == {"sales": 0}
            assert res["changes_made"]["missing_filled"] == 2
        finally:
            _pop(ex, res["data_handle"])


class TestFinerCandidateRejected:
    def test_minute_burst_inside_30min_data_is_not_expanded(self):
        # Three readings one minute apart would let pandas infer "min" from that
        # window; every timestamp sits on a minute boundary, so only the modal
        # interval check stops the series being blown up 30-fold.
        ex = get_executor()
        regular = pd.date_range("2024-01-01 00:00", periods=12, freq="30min")
        burst = pd.DatetimeIndex(["2024-01-01 06:01", "2024-01-01 06:02"])
        dates = [str(ts) for ts in regular.append(burst).sort_values()]
        values = [float(v) for v in range(len(dates))]
        res = _load(ex, dates, values)
        try:
            y = ex._data_handles[res["data_handle"]]["y"]
            assert len(y) == len(dates)
            assert list(y) == values
            assert not res["changes_made"]["frequency_set"]
            assert "does not align" in res["changes_made"]["frequency_warning"]
        finally:
            _pop(ex, res["data_handle"])
