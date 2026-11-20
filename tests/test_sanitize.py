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

    def test_index(self):
        result = sanitize_for_json(pd.period_range("1949-01", periods=3, freq="M"))
        assert isinstance(result, list) and len(result) == 3
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
        assert sanitize_for_json(np.float64("nan")) is None
        json.dumps(sanitize_for_json(np.float64("nan")), allow_nan=False)

    def test_circular_dict(self):
        d = {"a": 1}
        d["self"] = d
        result = sanitize_for_json(d)
        assert result["self"] == "<circular reference>"
        json.dumps(result)

    def test_circular_list(self):
        lst = [1, 2]
        lst.append(lst)
        result = sanitize_for_json(lst)
        assert result[2] == "<circular reference>"
        json.dumps(result)

    def test_indirect_cycle(self):
        a = {}
        b = {"a": a}
        a["b"] = b
        result = sanitize_for_json(a)
        assert result["b"]["a"] == "<circular reference>"
        json.dumps(result)

    def test_sibling_containers_not_falsely_flagged(self):
        shared = {"x": 1}
        result = sanitize_for_json({"left": shared, "right": shared})
        assert result == {"left": {"x": 1}, "right": {"x": 1}}

    def test_inf(self):
        assert sanitize_for_json(np.float64("inf")) is None
        assert sanitize_for_json(np.float64("-inf")) is None
        json.dumps(sanitize_for_json(np.float64("inf")), allow_nan=False)

    def test_nan_in_dict(self):
        result = sanitize_for_json({"score": np.float64("nan"), "n": 10})
        json.dumps(result, allow_nan=False)
        assert result == {"score": None, "n": 10}

    @pytest.mark.parametrize("number", [float("nan"), float("inf"), float("-inf")])
    def test_builtin_nonfinite_values_are_null(self, number):
        result = sanitize_for_json({"score": number, "nested": [number]})
        assert result == {"score": None, "nested": [None]}
        json.dumps(result, allow_nan=False)

    def test_array_frame_and_index_use_strict_json(self):
        result = sanitize_for_json(
            {
                "array": np.array([1.0, np.nan, np.inf]),
                "frame": pd.DataFrame({"x": [1.0, np.nan, -np.inf]}),
                "index": pd.Index([1.0, np.nan]),
            }
        )
        json.dumps(result, allow_nan=False)
        assert result["array"] == [1.0, None, None]
        assert result["frame"] == [{"x": 1.0}, {"x": None}, {"x": None}]
        assert result["index"] == [1.0, None]
