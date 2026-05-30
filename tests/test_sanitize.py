"""Tests for sanitize_for_json in the MCP server."""

import json

import numpy as np
import pandas as pd
import pytest

from sktime_mcp.server import sanitize_for_json


class TestNumpyScalars:
    """Sanitize NumPy scalar types to JSON-safe Python types."""

    @pytest.mark.parametrize(
        "value,expected",
        [
            (np.int8(42), 42),
            (np.int16(42), 42),
            (np.int32(42), 42),
            (np.int64(42), 42),
        ],
    )
    def test_integer_types(self, value, expected):
        assert sanitize_for_json(value) == expected

    @pytest.mark.parametrize(
        "value,expected",
        [
            (np.float32(3.14), float(np.float32(3.14))),
            (np.float64(0.95), 0.95),
        ],
    )
    def test_float_types(self, value, expected):
        assert sanitize_for_json(value) == expected

    def test_bool_types(self):
        assert sanitize_for_json(np.bool_(True)) is True
        assert sanitize_for_json(np.bool_(False)) is False


class TestNumpyArrays:
    """Sanitize NumPy arrays to JSON-safe lists."""

    def test_int_array(self):
        assert sanitize_for_json(np.array([1, 2, 3])) == [1, 2, 3]

    def test_float_array(self):
        result = sanitize_for_json(np.array([1.1, 2.2, 3.3]))
        assert result == [float(x) for x in [1.1, 2.2, 3.3]]

    def test_2d_array(self):
        result = sanitize_for_json(np.array([[1, 2], [3, 4]]))
        json.dumps(result)


class TestPandasTypes:
    """Sanitize Pandas types to JSON-safe equivalents."""

    def test_timestamp(self):
        assert sanitize_for_json(pd.Timestamp("2023-01-01")) == "2023-01-01T00:00:00"

    def test_nat(self):
        assert sanitize_for_json(pd.NaT) is None

    def test_na(self):
        assert sanitize_for_json(pd.NA) is None

    def test_series(self):
        result = sanitize_for_json(pd.Series([1, 2, 3]))
        json.dumps(result)

    def test_dataframe(self):
        df = pd.DataFrame({"value": [1.0, 2.0], "flag": [True, False]})
        result = sanitize_for_json(df)
        json.dumps(result)


class TestNestedToolOutput:
    """Realistic nested dicts like actual tool responses."""

    def test_full_tool_output(self):
        result = {
            "success": True,
            "predictions": {1: np.float64(450.2), 2: np.float64(460.5)},
            "horizon": np.int64(12),
            "fitted_at": pd.Timestamp("2023-01-01"),
            "residuals": np.array([0.1, -0.2, 0.05]),
            "metadata": {
                "dataset": "airline",
                "n_samples": np.int32(144),
                "has_exog": np.bool_(False),
            },
            "cv_results": [
                {"fold": np.int32(1), "score": np.float64(0.93)},
                {"fold": np.int32(2), "score": np.float64(0.89)},
            ],
        }
        sanitized = sanitize_for_json(result)
        json.dumps(sanitized)


class TestStandardPythonTypes:
    """Standard types pass through unchanged."""

    @pytest.mark.parametrize(
        "value,expected",
        [
            ("hello", "hello"),
            (42, 42),
            (3.14, 3.14),
            (True, True),
            (None, None),
        ],
    )
    def test_scalars(self, value, expected):
        assert sanitize_for_json(value) == expected

    def test_list_of_mixed(self):
        result = sanitize_for_json([1, "two", None, True])
        json.dumps(result)

    def test_nested_dict(self):
        result = sanitize_for_json({"a": {"b": 1}})
        json.dumps(result)


class TestEdgeCases:
    """Edge cases and boundary conditions."""

    def test_empty_list(self):
        assert sanitize_for_json([]) == []

    def test_empty_dict(self):
        assert sanitize_for_json({}) == {}

    def test_nan(self):
        result = sanitize_for_json(np.float64("nan"))
        json.dumps(result, default=str)
