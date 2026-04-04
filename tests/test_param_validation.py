"""
Tests for parameter validation in instantiate_estimator tool.

Covers Issue #3: [ENH] Add type validation for params in instantiate_estimator.
"""

import sys

import pytest

sys.path.insert(0, "src")

from sktime_mcp.tools.instantiate import (
    _validate_params,
    instantiate_estimator_tool,
    instantiate_pipeline_tool,
)


class TestValidateParams:
    """Tests for the _validate_params helper function."""

    def test_params_none_is_valid(self):
        """None params should be valid (use estimator defaults)."""
        result = _validate_params(None)
        assert result["valid"] is True
        assert result["warnings"] == []

    def test_params_empty_dict_valid(self):
        """Empty dict params should be valid."""
        result = _validate_params({})
        assert result["valid"] is True

    def test_params_valid_dict(self):
        """A normal dict with primitive values should be valid."""
        result = _validate_params({"order": [1, 1, 1], "suppress_warnings": True})
        assert result["valid"] is True

    def test_params_string_rejected(self):
        """String passed as params should be rejected."""
        result = _validate_params("invalid")
        assert result["valid"] is False
        assert "must be a dictionary" in result["error"]

    def test_params_list_rejected(self):
        """List passed as params should be rejected."""
        result = _validate_params([1, 2, 3])
        assert result["valid"] is False
        assert "must be a dictionary" in result["error"]

    def test_params_int_rejected(self):
        """Integer passed as params should be rejected."""
        result = _validate_params(42)
        assert result["valid"] is False
        assert "must be a dictionary" in result["error"]

    def test_params_callable_value_rejected(self):
        """Dict with callable value should be rejected."""
        result = _validate_params({"fn": lambda: None})
        assert result["valid"] is False
        assert "Unsupported type" in result["error"]
        assert "fn" in result["error"]

    def test_params_class_value_rejected(self):
        """Dict with class/type value should be rejected."""
        result = _validate_params({"cls": object})
        assert result["valid"] is False
        assert "Unsupported type" in result["error"]

    def test_params_nested_unsafe_value_rejected(self):
        """Nested callable inside a list should be rejected."""
        result = _validate_params({"items": [1, 2, lambda: None]})
        assert result["valid"] is False
        assert "Unsupported type" in result["error"]

    def test_unknown_key_produces_warning(self):
        """Unknown param key should pass validation but produce a warning."""
        result = _validate_params(
            {"nonexistent_param_xyz": 1},
            estimator_name="NaiveForecaster",
        )
        assert result["valid"] is True
        assert len(result["warnings"]) > 0
        assert "nonexistent_param_xyz" in result["warnings"][0]


class TestInstantiateEstimatorValidation:
    """Tests for validation in the instantiate_estimator_tool."""

    def test_valid_params_succeed(self):
        """Valid params with a real estimator should succeed."""
        result = instantiate_estimator_tool("NaiveForecaster", {"strategy": "last"})
        assert result["success"] is True
        assert "handle" in result

    def test_invalid_type_returns_error(self):
        """Non-dict params should return success=False with error."""
        result = instantiate_estimator_tool("NaiveForecaster", "invalid")
        assert result["success"] is False
        assert "must be a dictionary" in result["error"]

    def test_unsafe_value_returns_error(self):
        """Callable param value should return success=False with error."""
        result = instantiate_estimator_tool("NaiveForecaster", {"fn": print})
        assert result["success"] is False
        assert "Unsupported type" in result["error"]


class TestPipelineParamsValidation:
    """Tests for validation in the instantiate_pipeline_tool."""

    def test_pipeline_invalid_params_list_type(self):
        """Non-list params_list should return error."""
        result = instantiate_pipeline_tool(["NaiveForecaster"], "not_a_list")
        assert result["success"] is False
        assert "params_list" in result["error"]

    def test_pipeline_invalid_param_dict_in_list(self):
        """Non-dict entry in params_list should return error."""
        result = instantiate_pipeline_tool(["NaiveForecaster"], ["not_a_dict"])
        assert result["success"] is False
        assert "must be a dictionary" in result["error"]

    def test_pipeline_unsafe_value_in_params_list(self):
        """Callable value in pipeline params should return error."""
        result = instantiate_pipeline_tool(
            ["NaiveForecaster"],
            [{"fn": lambda: None}],
        )
        assert result["success"] is False
        assert "Unsupported type" in result["error"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
