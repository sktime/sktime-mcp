"""
Tests for parameter validation in instantiate_estimator tool.

Covers Issue #3: [ENH] Add type validation for params in instantiate_estimator.
"""

import sys

import pytest

sys.path.insert(0, "src")

from sktime_mcp.tools.fit_predict import predict_tool
from sktime_mcp.tools.instantiate import (
    _validate_components,
    _validate_estimator_name,
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

    @pytest.mark.parametrize(
        ("invalid_params", "expected_error"),
        [
            ("invalid", "must be a dictionary"),
            ([1, 2, 3], "must be a dictionary"),
            (42, "must be a dictionary"),
            ({"fn": lambda: None}, "Unsupported type"),
            ({"cls": object}, "Unsupported type"),
            ({"items": [1, 2, lambda: None]}, "Unsupported type"),
        ],
    )
    def test_params_invalid_inputs_rejected(self, invalid_params, expected_error):
        """Tests to reject invalid input parameters"""
        result = _validate_params(invalid_params)
        assert result["valid"] is False
        assert expected_error in result["error"]

    def test_unknown_key_produces_warning(self):
        """Unknown param key should pass validation but produce a warning."""
        result = _validate_params(
            {"nonexistent_param_xyz": 1},
            estimator_name="NaiveForecaster",
        )
        assert result["valid"] is True
        assert len(result["warnings"]) > 0
        assert "nonexistent_param_xyz" in result["warnings"][0]


class TestValidateTopLevelInputs:
    """Tests for top-level estimator/components validation helpers."""

    def test_estimator_name_must_be_string(self):
        """Estimator name should reject non-string values."""
        result = _validate_estimator_name(["ARIMA"])
        assert result["valid"] is False
        assert "'estimator' must be a non-empty string" in result["error"]

    def test_estimator_name_rejects_blank_string(self):
        """Estimator name should reject blank strings."""
        result = _validate_estimator_name("   ")
        assert result["valid"] is False
        assert "non-empty string" in result["error"]

    def test_components_must_be_list(self):
        """Components should reject non-list values."""
        result = _validate_components("ARIMA")
        assert result["valid"] is False
        assert "'components' must be a non-empty list" in result["error"]

    def test_components_reject_nested_non_string(self):
        """Components should reject nested non-string entries."""
        result = _validate_components(["ARIMA", ["NaiveForecaster"]])
        assert result["valid"] is False
        assert "components[1]" in result["error"]


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

    def test_invalid_estimator_type_returns_error(self):
        """Non-string estimator names should return success=False with error."""
        result = instantiate_estimator_tool(["NaiveForecaster"])
        assert result["success"] is False
        assert "'estimator' must be a non-empty string" in result["error"]


class TestPipelineParamsValidation:
    """Tests for validation in the instantiate_pipeline_tool."""

    def test_pipeline_components_must_be_list(self):
        """String components should return an explicit type validation error."""
        result = instantiate_pipeline_tool("NaiveForecaster")
        assert result["success"] is False
        assert "'components' must be a non-empty list" in result["error"]

    def test_pipeline_components_reject_non_string_entries(self):
        """Nested non-string components should return success=False with error."""
        result = instantiate_pipeline_tool(["NaiveForecaster", ["ARIMA"]])
        assert result["success"] is False
        assert "components[1]" in result["error"]

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


class TestFitPredictValidation:
    """Tests for parameter validation in fit_predict tools."""

    @pytest.mark.parametrize(
        "invalid_horizon, expected_error",
        [
            ("five", "must be an integer"),
            (0, "greater than 0"),
            (-3, "greater than 0"),
            (None, "must be an integer"),
            (3.14, "must be an integer"),
            ([1, 2], "must be an integer"),
        ],
    )
    def test_predict_tool_horizon_string(self, invalid_horizon, expected_error):
        """Invalid horizons should be rejected with the correct error"""
        result = predict_tool("fake_handle", horizon=invalid_horizon)
        assert result["success"] is False
        assert expected_error in result["error"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
