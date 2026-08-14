"""transform/convert dead ends and JSON round-trip (#537)."""

import contextlib
import os
import tempfile

import pandas as pd
import pytest

from sktime_mcp.runtime.executor import get_executor
from sktime_mcp.tools.save_data import save_data_tool
from sktime_mcp.tools.transform_data import transform_data_tool


@pytest.fixture
def series_handle():
    ex = get_executor()
    res = ex.load_data_source(
        {
            "type": "pandas",
            "data": {
                "date": [f"2024-{m:02d}-01" for m in range(1, 13)],
                "value": [float(i) for i in range(12)],
            },
            "time_column": "date",
            "target_column": "value",
        }
    )
    dh = res["data_handle"]
    yield ex, dh
    for h in list(ex._data_handles):
        if h == dh:
            ex._data_handles.pop(h, None)


class TestConvertDeadEnds:
    def test_ndarray_target_rejected(self, series_handle):
        ex, dh = series_handle
        res = transform_data_tool(data_handle=dh, action="convert", to_mtype="np.ndarray")
        assert not res["success"]
        assert "no time index" in res["error"].lower()

    def test_series_to_panel_clean_error(self, series_handle):
        ex, dh = series_handle
        res = transform_data_tool(data_handle=dh, action="convert", to_mtype="pd-multiindex")
        assert not res["success"]
        # clean domain message, not a multi-paragraph mtype dump
        assert "scitype" in res["error"].lower() or "panel" in res["error"].lower()
        assert "No valid mtype" not in res["error"]

    def test_valid_convert_still_works(self, series_handle):
        ex, dh = series_handle
        res = transform_data_tool(data_handle=dh, action="convert", to_mtype="pd.DataFrame")
        assert res["success"], res


class TestDuplicateTimestamps:
    def test_duplicates_load_and_dedup(self):
        ex = get_executor()
        res = ex.load_data_source(
            {
                "type": "pandas",
                "data": {
                    "date": ["2024-01-01", "2024-01-01", "2024-02-01", "2024-03-01"],
                    "value": [1.0, 1.5, 2.0, 3.0],
                },
                "time_column": "date",
                "target_column": "value",
            }
        )
        try:
            # previously this hard-failed with "Duplicate time indices found"
            assert res["success"], res
            # auto-format removed the duplicate
            handle = res["data_handle"]
            assert not ex._data_handles[handle]["y"].index.duplicated().any()
        finally:
            ex._data_handles.pop(res.get("data_handle"), None)


class TestJsonRoundTrip:
    def test_json_save_then_load(self, series_handle):
        ex, dh = series_handle
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "data.json")
            saved = save_data_tool(dh, path=path, format="json")
            assert saved["success"], saved
            loaded = ex.load_data_source(
                {"type": "file", "path": path, "time_column": "time", "target_column": "value"}
            )
            try:
                assert loaded["success"], loaded
            finally:
                ex._data_handles.pop(loaded.get("data_handle"), None)
