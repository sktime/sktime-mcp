"""Tests for the plot_series_tool."""

import base64
import os
import tempfile

import pandas as pd
import pytest

from sktime_mcp.tools.plotting import (
    _coerce_indices,
    _reconcile_labels,
    plot_series_tool,
)
from sktime_mcp.runtime.executor import get_executor


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_handles():
    """Clear data handles before and after each test."""
    executor = get_executor()
    executor._data_handles.clear()
    yield
    executor._data_handles.clear()


def _register_series(handle_id: str, series: pd.Series) -> None:
    """Register a series under the given handle in the executor."""
    executor = get_executor()
    executor._data_handles[handle_id] = {"y": series}


def _make_airline_like(n: int = 24) -> pd.Series:
    """Create a small airline-like monthly series."""
    idx = pd.date_range("2020-01", periods=n, freq="MS")
    return pd.Series(range(100, 100 + n), index=idx, name="Passengers")


# ---------------------------------------------------------------------------
# Unit tests — helper functions
# ---------------------------------------------------------------------------


class TestReconcileLabels:
    def test_none_returns_none(self):
        assert _reconcile_labels(None, 3) is None

    def test_matching_length(self):
        labels = ["a", "b", "c"]
        assert _reconcile_labels(labels, 3) == ["a", "b", "c"]

    def test_too_few_labels_padded(self):
        result = _reconcile_labels(["train"], 3)
        assert len(result) == 3
        assert result[0] == "train"
        assert result[1] == "Series 1"
        assert result[2] == "Series 2"

    def test_too_many_labels_truncated(self):
        result = _reconcile_labels(["a", "b", "c", "d"], 2)
        assert result == ["a", "b"]


class TestCoerceIndices:
    def test_uniform_datetime_no_change(self):
        s1 = _make_airline_like(12)
        s2 = _make_airline_like(12)
        result = _coerce_indices([s1, s2])
        assert all(isinstance(s.index, pd.DatetimeIndex) for s in result)

    def test_mixed_period_datetime_coerced(self):
        dt_series = _make_airline_like(12)
        period_series = dt_series.copy()
        period_series.index = dt_series.index.to_period("M")
        result = _coerce_indices([dt_series, period_series])
        assert all(isinstance(s.index, pd.DatetimeIndex) for s in result)


# ---------------------------------------------------------------------------
# Integration tests — plot_series_tool
# ---------------------------------------------------------------------------


class TestPlotSeriesTool:
    """Integration tests for the full plot_series_tool."""

    def test_handle_not_found(self):
        result = plot_series_tool(data_handles=["nonexistent"])
        assert result["success"] is False
        assert "not found" in result["error"]
        assert "available_handles" in result

    def test_unsupported_format(self):
        _register_series("s1", _make_airline_like())
        result = plot_series_tool(
            data_handles=["s1"], image_format="bmp"
        )
        assert result["success"] is False
        assert "Unsupported image format" in result["error"]

    def test_single_series_base64(self):
        _register_series("train", _make_airline_like())
        result = plot_series_tool(data_handles=["train"])
        assert result["success"] is True
        assert "image_base64" in result
        # Verify it's valid base64
        raw = base64.b64decode(result["image_base64"])
        assert len(raw) > 0

    def test_single_series_metadata(self):
        _register_series("train", _make_airline_like())
        result = plot_series_tool(
            data_handles=["train"],
            title="Test Plot",
            figsize=[8, 4],
            dpi=100,
        )
        assert result["success"] is True
        assert result["n_series"] == 1
        assert result["figsize"] == [8, 4]
        assert result["dpi"] == 100
        assert result["image_format"] == "png"

    def test_multiple_series(self):
        _register_series("s1", _make_airline_like(12))
        _register_series("s2", _make_airline_like(12))
        result = plot_series_tool(
            data_handles=["s1", "s2"],
            labels=["Train", "Test"],
        )
        assert result["success"] is True
        assert result["n_series"] == 2
        assert result["labels_used"] == ["Train", "Test"]

    def test_save_to_file(self):
        _register_series("s1", _make_airline_like())
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(tmpdir, "test_plot.png")
            result = plot_series_tool(
                data_handles=["s1"],
                path=out_path,
            )
            assert result["success"] is True
            assert result["path"] == out_path
            assert os.path.exists(out_path)
            assert os.path.getsize(out_path) > 0

    def test_save_svg_format(self):
        _register_series("s1", _make_airline_like())
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(tmpdir, "test_plot.svg")
            result = plot_series_tool(
                data_handles=["s1"],
                path=out_path,
                image_format="svg",
            )
            assert result["success"] is True
            assert result["image_format"] == "svg"

    def test_label_mismatch_handled(self):
        """Labels that don't match series count should not crash."""
        _register_series("s1", _make_airline_like())
        _register_series("s2", _make_airline_like())
        # Fewer labels than series
        result = plot_series_tool(
            data_handles=["s1", "s2"],
            labels=["Only One"],
        )
        assert result["success"] is True
        assert len(result["labels_used"]) == 2

    def test_custom_axis_labels(self):
        _register_series("s1", _make_airline_like())
        result = plot_series_tool(
            data_handles=["s1"],
            x_label="Date",
            y_label="Passengers (thousands)",
            title="Airline Traffic",
        )
        assert result["success"] is True

    def test_markers_param(self):
        _register_series("s1", _make_airline_like(12))
        result = plot_series_tool(
            data_handles=["s1"],
            markers="o",
        )
        assert result["success"] is True

    def test_mixed_handles_partial_missing(self):
        """When some handles exist and some don't, report all missing."""
        _register_series("real", _make_airline_like())
        result = plot_series_tool(
            data_handles=["real", "fake1", "fake2"]
        )
        assert result["success"] is False
        assert "fake1" in result["error"]
        assert "fake2" in result["error"]
