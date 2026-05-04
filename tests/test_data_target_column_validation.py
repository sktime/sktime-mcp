"""Tests for explicit target column validation in data adapters."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sktime_mcp.data.adapters.pandas_adapter import PandasAdapter
from sktime_mcp.runtime.executor import Executor


def test_to_sktime_format_rejects_missing_explicit_target_column():
    """An explicit target_column typo should fail instead of silently using column 0."""
    adapter = PandasAdapter(
        {
            "type": "pandas",
            "data": {
                "date": ["2020-01-01", "2020-01-02", "2020-01-03"],
                "value": [1, 2, 3],
            },
            "time_column": "date",
            "target_column": "sales",
        }
    )
    data = adapter.load()

    with pytest.raises(ValueError, match="Target column 'sales' not found"):
        adapter.to_sktime_format(data)


def test_load_data_source_returns_error_for_missing_explicit_target_column():
    """Executor should surface the missing target column as a structured tool error."""
    executor = Executor()

    result = executor.load_data_source(
        {
            "type": "pandas",
            "data": {
                "date": ["2020-01-01", "2020-01-02", "2020-01-03"],
                "value": [1, 2, 3],
            },
            "time_column": "date",
            "target_column": "sales",
        }
    )

    assert result["success"] is False
    assert result["error_type"] == "ValueError"
    assert "Target column 'sales' not found" in result["error"]
    assert "value" in result["error"]

