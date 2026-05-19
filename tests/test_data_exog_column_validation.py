"""Tests for explicit exogenous column validation in data adapters."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sktime_mcp.data.adapters.pandas_adapter import PandasAdapter
from sktime_mcp.runtime.executor import Executor


def test_to_sktime_format_rejects_missing_explicit_exog_column():
    """Explicit exog_columns should fail if any requested column is absent."""
    adapter = PandasAdapter(
        {
            "type": "pandas",
            "data": {
                "date": ["2020-01-01", "2020-01-02", "2020-01-03"],
                "value": [1, 2, 3],
                "promo": [0, 1, 0],
            },
            "time_column": "date",
            "target_column": "value",
            "exog_columns": ["promo", "holiday"],
        }
    )
    data = adapter.load()

    with pytest.raises(ValueError, match="Exogenous column\\(s\\) not found"):
        adapter.to_sktime_format(data)


def test_load_data_source_returns_error_for_missing_explicit_exog_column():
    """Executor should surface missing exog columns as a structured tool error."""
    executor = Executor()

    result = executor.load_data_source(
        {
            "type": "pandas",
            "data": {
                "date": ["2020-01-01", "2020-01-02", "2020-01-03"],
                "value": [1, 2, 3],
                "promo": [0, 1, 0],
            },
            "time_column": "date",
            "target_column": "value",
            "exog_columns": ["promo", "holiday"],
        }
    )

    assert result["success"] is False
    assert result["error_type"] == "ValueError"
    assert "Exogenous column(s) not found" in result["error"]
    assert "holiday" in result["error"]
    assert "promo" in result["error"]


def test_to_sktime_format_keeps_all_valid_explicit_exog_columns():
    """Valid exog_columns should be preserved exactly when all columns exist."""
    adapter = PandasAdapter(
        {
            "type": "pandas",
            "data": {
                "date": ["2020-01-01", "2020-01-02", "2020-01-03"],
                "value": [1, 2, 3],
                "promo": [0, 1, 0],
                "price": [10, 11, 12],
            },
            "time_column": "date",
            "target_column": "value",
            "exog_columns": ["promo", "price"],
        }
    )
    data = adapter.load()

    y, X = adapter.to_sktime_format(data)

    assert y.name == "value"
    assert list(X.columns) == ["promo", "price"]

