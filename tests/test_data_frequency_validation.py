"""Tests for explicit frequency validation in data adapters."""

from pathlib import Path

import pytest

from sktime_mcp.data.adapters.file_adapter import FileAdapter
from sktime_mcp.data.adapters.pandas_adapter import PandasAdapter


def test_pandas_adapter_rejects_invalid_explicit_frequency():
    """Invalid user-provided frequency should fail instead of being ignored."""
    adapter = PandasAdapter(
        {
            "type": "pandas",
            "data": {
                "date": ["2020-01-01", "2020-01-02", "2020-01-03"],
                "value": [1, 2, 3],
            },
            "time_column": "date",
            "target_column": "value",
            "frequency": "not_a_freq",
        }
    )

    with pytest.raises(ValueError, match="Invalid frequency 'not_a_freq'"):
        adapter.load()


def test_file_adapter_rejects_invalid_explicit_frequency(tmp_path: Path):
    """Invalid explicit frequency should fail for file-backed data too."""
    path = tmp_path / "series.csv"
    path.write_text("date,value\n2020-01-01,1\n2020-01-02,2\n2020-01-03,3\n")

    adapter = FileAdapter(
        {
            "type": "file",
            "path": str(path),
            "time_column": "date",
            "target_column": "value",
            "frequency": "not_a_freq",
        }
    )

    with pytest.raises(ValueError, match="Invalid frequency 'not_a_freq'"):
        adapter.load()


def test_pandas_adapter_applies_valid_explicit_frequency():
    """Valid explicit frequency should still be applied normally."""
    adapter = PandasAdapter(
        {
            "type": "pandas",
            "data": {
                "date": ["2020-01-01", "2020-01-03"],
                "value": [1, 3],
            },
            "time_column": "date",
            "target_column": "value",
            "frequency": "D",
        }
    )

    data = adapter.load()

    assert len(data) == 3
    assert data.index.freqstr == "D"
