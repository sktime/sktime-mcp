"""Tests for the get_estimator_params MCP tool."""

import pytest

from sktime_mcp.tools.inspect_estimator import (
    _extract_fitted_params,
    _to_json_safe,
    get_estimator_params_tool,
)
from sktime_mcp.tools.instantiate import instantiate_estimator_tool


class TestGetEstimatorParamsTool:
    """Test suite for get_estimator_params_tool."""

    def test_returns_hyperparameters_for_unfitted_estimator(self):
        """Unfitted estimator should return hyperparameters but no fitted_params."""
        inst = instantiate_estimator_tool("NaiveForecaster")
        assert inst["success"]
        handle = inst["handle"]

        result = get_estimator_params_tool(handle)
        assert result["success"] is True
        assert result["handle"] == handle
        assert result["estimator_name"] == "NaiveForecaster"
        assert result["is_fitted"] is False
        assert "hyperparameters" in result
        assert "created_at" in result
        # No fitted_params for unfitted estimator
        assert "fitted_params" not in result

    def test_returns_fitted_params_after_fitting(self):
        """After fit_predict, fitted_params should be populated."""
        from sktime_mcp.tools.fit_predict import fit_predict_tool

        inst = instantiate_estimator_tool("NaiveForecaster")
        assert inst["success"]
        handle = inst["handle"]

        fp = fit_predict_tool(handle, "airline", horizon=3)
        assert fp["success"]

        result = get_estimator_params_tool(handle)
        assert result["success"] is True
        assert result["is_fitted"] is True
        assert "fitted_params" in result

    def test_invalid_handle_returns_error(self):
        """Non-existent handle should fail gracefully."""
        result = get_estimator_params_tool("est_nonexistent")
        assert result["success"] is False
        assert "error" in result
        assert "Handle not found" in result["error"]

    def test_empty_handle_returns_error(self):
        """Empty string handle should fail with a clear message."""
        result = get_estimator_params_tool("")
        assert result["success"] is False
        assert "non-empty string" in result["error"]

    def test_non_string_handle_returns_error(self):
        """Non-string handle should fail."""
        result = get_estimator_params_tool(123)
        assert result["success"] is False
        assert "non-empty string" in result["error"]


class TestToJsonSafe:
    """Test suite for the _to_json_safe helper."""

    def test_primitives(self):
        assert _to_json_safe(42) == 42
        assert _to_json_safe(3.14) == 3.14
        assert _to_json_safe("hello") == "hello"
        assert _to_json_safe(True) is True
        assert _to_json_safe(None) is None

    def test_dict(self):
        result = _to_json_safe({"a": 1, "b": [2, 3]})
        assert result == {"a": 1, "b": [2, 3]}

    def test_list(self):
        result = _to_json_safe([1, "two", 3.0])
        assert result == [1, "two", 3.0]

    def test_numpy_scalar(self):
        np = pytest.importorskip("numpy")
        assert _to_json_safe(np.int64(42)) == 42
        assert _to_json_safe(np.float64(3.14)) == 3.14

    def test_numpy_array_small(self):
        np = pytest.importorskip("numpy")
        arr = np.array([1, 2, 3])
        assert _to_json_safe(arr) == [1, 2, 3]

    def test_numpy_array_large(self):
        np = pytest.importorskip("numpy")
        arr = np.zeros(100)
        result = _to_json_safe(arr)
        assert isinstance(result, str)
        assert "ndarray" in result

    def test_fallback_to_str(self):
        """Unsupported types should fall back to str()."""

        class Foo:
            def __str__(self):
                return "Foo()"

        assert _to_json_safe(Foo()) == "Foo()"


class TestExtractFittedParams:
    """Test suite for _extract_fitted_params helper."""

    def test_extracts_trailing_underscore_attrs(self):
        """Should capture public attributes ending with _."""

        class FakeEstimator:
            aic_ = 450.2
            order_ = (1, 1, 1)
            _private_ = "hidden"

            def some_method_(self):
                pass

        params = _extract_fitted_params(FakeEstimator())
        assert "aic_" in params
        assert params["aic_"] == 450.2
        assert "order_" in params
        # Private attributes should be excluded
        assert "_private_" not in params
