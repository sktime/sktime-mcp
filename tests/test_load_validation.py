"""load_data_source validation gaps (#533).

- NB-07: a non-numeric target must be flagged (not silently valid).
- NB-06: a 1–2 row series must load, not crash on frequency inference.
- NB-11: a bad file time_column must give a clean "not found" message,
  not a leaked pandas parse_dates internal.
"""

import csv
import os
import tempfile

import pytest

from sktime_mcp.runtime.executor import get_executor


class TestNonNumericTarget:
    def test_object_dtype_target_warns(self):
        ex = get_executor()
        res = ex.load_data_source(
            {
                "type": "pandas",
                "data": {
                    "date": ["2024-01-01", "2024-02-01", "2024-03-01"],
                    "value": ["1", "2", "3"],  # strings -> object dtype
                },
                "time_column": "date",
                "target_column": "value",
            }
        )
        try:
            assert res["success"]  # still loads
            warnings = res["validation"]["warnings"]
            assert any("non-numeric" in w.lower() and "value" in w for w in warnings), warnings
        finally:
            ex._data_handles.pop(res.get("data_handle"), None)

    def test_numeric_target_no_warning(self):
        ex = get_executor()
        res = ex.load_data_source(
            {
                "type": "pandas",
                "data": {
                    "date": ["2024-01-01", "2024-02-01", "2024-03-01"],
                    "value": [1.0, 2.0, 3.0],
                },
                "time_column": "date",
                "target_column": "value",
            }
        )
        try:
            assert not any("non-numeric" in w.lower() for w in res["validation"]["warnings"])
        finally:
            ex._data_handles.pop(res.get("data_handle"), None)


class TestShortSeries:
    @pytest.mark.parametrize("n", [1, 2])
    def test_short_series_loads(self, n):
        ex = get_executor()
        res = ex.load_data_source(
            {
                "type": "pandas",
                "data": {
                    "date": ["2024-01-01", "2024-02-01"][:n],
                    "value": [10.0, 20.0][:n],
                },
                "time_column": "date",
                "target_column": "value",
            }
        )
        try:
            assert res["success"], res
        finally:
            ex._data_handles.pop(res.get("data_handle"), None)


class TestFileBadTimeColumn:
    def test_missing_time_column_clean_error(self):
        ex = get_executor()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "data.csv")
            with open(path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["timestamp", "value"])
                for i in range(5):
                    w.writerow([f"2024-01-0{i + 1}", i])
            res = ex.load_data_source(
                {"type": "file", "path": path, "time_column": "index", "target_column": "value"}
            )
        assert not res["success"]
        err = res["error"]
        assert "not found" in err.lower()
        assert "parse_dates" not in err  # no leaked pandas internal
        assert "timestamp" in err  # lists available columns

    def test_missing_target_column_lists_available(self):
        ex = get_executor()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "data.csv")
            with open(path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["timestamp", "value"])
                for i in range(5):
                    w.writerow([f"2024-01-0{i + 1}", i])
            res = ex.load_data_source(
                {"type": "file", "path": path, "time_column": "timestamp", "target_column": "nope"}
            )
        assert not res["success"]
        assert "nope" in res["error"]
