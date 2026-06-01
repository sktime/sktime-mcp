"""Tests for the new Data Management tools: inspect_data, split_data, transform_data, save_data."""

import os
import tempfile
from pathlib import Path

import pandas as pd

from sktime_mcp.runtime.executor import Executor
from sktime_mcp.tools.inspect_data import inspect_data_tool
from sktime_mcp.tools.save_data import save_data_tool
from sktime_mcp.tools.split_data import split_data_tool
from sktime_mcp.tools.transform_data import transform_data_tool


def _make_executor_with_data():
    """Create an Executor and load the airline demo dataset into a handle."""
    executor = Executor()
    # Load a small demo dataset (airline)
    result = executor.load_data_source(
        {
            "type": "pandas",
            "data": {
                "date": pd.date_range("2020-01", periods=60, freq="MS")
                .strftime("%Y-%m-%d")
                .tolist(),
                "value": list(range(60)),
            },
            "time_column": "date",
            "target_column": "value",
        }
    )
    assert result["success"], f"Setup failed: {result}"
    return executor, result["data_handle"]


# ───────────────────────────────────────────────────────────────────────────
# inspect_data
# ───────────────────────────────────────────────────────────────────────────


class TestInspectData:
    def test_inspect_returns_all_fields(self):
        executor, handle = _make_executor_with_data()
        # Patch singleton so the tool sees our executor
        import sktime_mcp.tools.inspect_data as mod

        _orig = mod.get_executor
        mod.get_executor = lambda: executor
        try:
            result = inspect_data_tool(handle)
        finally:
            mod.get_executor = _orig

        assert result["success"]
        for key in [
            "mtype",
            "scitype",
            "shape",
            "columns",
            "dtypes",
            "index_names",
            "freq",
            "cutoff",
            "n_missing",
            "head",
            "summary_stats",
        ]:
            assert key in result, f"Missing key: {key}"

        assert result["shape"][0] == 60
        assert result["n_missing"] == 0

    def test_inspect_unknown_handle(self):
        result = inspect_data_tool("data_nonexistent")
        assert not result["success"]
        assert "not found" in result["error"]


# ───────────────────────────────────────────────────────────────────────────
# split_data
# ───────────────────────────────────────────────────────────────────────────


class TestSplitData:
    def _patch(self, executor):
        import sktime_mcp.tools.split_data as mod

        self._orig = mod.get_executor
        mod.get_executor = lambda: executor

    def _unpatch(self):
        import sktime_mcp.tools.split_data as mod

        mod.get_executor = self._orig

    def test_split_by_test_size(self):
        executor, handle = _make_executor_with_data()
        self._patch(executor)
        try:
            result = split_data_tool(handle, test_size=0.2)
        finally:
            self._unpatch()

        assert result["success"]
        assert "train_handle" in result
        assert "test_handle" in result
        assert "cutoff" in result
        assert result["train_size"] + result["test_size"] == 60

    def test_split_by_fh(self):
        executor, handle = _make_executor_with_data()
        self._patch(executor)
        try:
            result = split_data_tool(handle, fh=12)
        finally:
            self._unpatch()

        assert result["success"]
        assert result["test_size"] == 12
        assert result["train_size"] == 48

    def test_split_requires_one_arg(self):
        executor, handle = _make_executor_with_data()
        self._patch(executor)
        try:
            # Neither
            r1 = split_data_tool(handle)
            assert not r1["success"]
            # Both
            r2 = split_data_tool(handle, test_size=0.2, fh=12)
            assert not r2["success"]
        finally:
            self._unpatch()

    def test_split_invalid_test_size(self):
        executor, handle = _make_executor_with_data()
        self._patch(executor)
        try:
            result = split_data_tool(handle, test_size=1.5)
        finally:
            self._unpatch()
        assert not result["success"]

    def test_split_unknown_handle(self):
        result = split_data_tool("data_nonexistent", test_size=0.2)
        assert not result["success"]


# ───────────────────────────────────────────────────────────────────────────
# transform_data
# ───────────────────────────────────────────────────────────────────────────


class TestTransformData:
    def _patch(self, executor):
        import sktime_mcp.tools.transform_data as mod

        self._orig = mod.get_executor
        mod.get_executor = lambda: executor

    def _unpatch(self):
        import sktime_mcp.tools.transform_data as mod

        mod.get_executor = self._orig

    def test_format_action(self):
        executor, handle = _make_executor_with_data()
        self._patch(executor)
        try:
            result = transform_data_tool(handle, action="format")
        finally:
            self._unpatch()

        assert result["success"]
        assert "data_handle" in result
        assert "changes_applied" in result
        assert isinstance(result["changes_applied"], list)

    def test_convert_requires_to_mtype(self):
        executor, handle = _make_executor_with_data()
        self._patch(executor)
        try:
            result = transform_data_tool(handle, action="convert")
        finally:
            self._unpatch()
        assert not result["success"]
        assert "to_mtype" in result["error"]

    def test_invalid_action(self):
        executor, handle = _make_executor_with_data()
        self._patch(executor)
        try:
            result = transform_data_tool(handle, action="bogus")
        finally:
            self._unpatch()
        assert not result["success"]
        assert "Unknown action" in result["error"]

    def test_unknown_handle(self):
        result = transform_data_tool("data_nonexistent", action="format")
        assert not result["success"]


# ───────────────────────────────────────────────────────────────────────────
# save_data
# ───────────────────────────────────────────────────────────────────────────


class TestSaveData:
    def _patch(self, executor):
        import sktime_mcp.tools.save_data as mod

        self._orig = mod.get_executor
        mod.get_executor = lambda: executor

    def _unpatch(self):
        import sktime_mcp.tools.save_data as mod

        mod.get_executor = self._orig

    def test_save_csv(self):
        executor, handle = _make_executor_with_data()
        self._patch(executor)
        try:
            with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
                path = f.name
            result = save_data_tool(handle, path=path, format="csv")
        finally:
            self._unpatch()

        assert result["success"]
        assert result["format"] == "csv"
        assert result["rows"] == 60
        assert Path(result["saved_path"]).exists()
        Path(path).unlink()

    def test_save_json(self):
        executor, handle = _make_executor_with_data()
        self._patch(executor)
        try:
            with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
                path = f.name
            result = save_data_tool(handle, path=path, format="json")
        finally:
            self._unpatch()

        assert result["success"]
        assert result["format"] == "json"
        Path(path).unlink()

    def test_save_unsupported_format(self):
        executor, handle = _make_executor_with_data()
        self._patch(executor)
        try:
            result = save_data_tool(handle, path="/tmp/out.xlsx", format="xlsx")
        finally:
            self._unpatch()
        assert not result["success"]
        assert "Unsupported format" in result["error"]

    def test_save_unknown_handle(self):
        result = save_data_tool("data_nonexistent", path="/tmp/out.csv")
        assert not result["success"]
