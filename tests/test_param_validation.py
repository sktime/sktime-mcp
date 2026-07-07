"""
Tests for parameter validation in instantiate_estimator tool.

Covers Issue #3: [ENH] Add type validation for params in instantiate_estimator.
"""

import sys

import pytest

sys.path.insert(0, "src")

from sktime_mcp.tools.fit_predict import predict_tool
from sktime_mcp.tools.instantiate import instantiate_estimator_tool


class TestInstantiateEstimatorValidation:
    """Tests for validation in the instantiate_estimator_tool."""

    def test_valid_spec_succeeds(self):
        """Valid spec with a real estimator should succeed."""
        result = instantiate_estimator_tool(spec="NaiveForecaster(strategy='last')")
        assert result["success"] is True
        assert "handle" in result

    def test_invalid_type_returns_error(self):
        """Non-string spec should return success=False with error."""
        result = instantiate_estimator_tool(spec=123)
        assert result["success"] is False
        assert "valid 'spec' string" in result["error"]

    def test_unsafe_value_returns_error(self):
        """Invalid spec string should return success=False with error from craft."""
        result = instantiate_estimator_tool(spec="NotARealEstimator()")
        assert result["success"] is False
        assert "error" in result

    def test_requires_spec(self):
        """Must provide a spec."""
        result = instantiate_estimator_tool(spec=None)
        assert result["success"] is False
        assert "required" in result["error"].lower()


class TestPipelineParamsValidation:
    """Tests for pipeline validation via the unified instantiate_estimator_tool."""

    def test_pipeline_composition_check(self):
        """Invalid pipeline composition (e.g. chaining forecasters) should fail validation."""
        result = instantiate_estimator_tool(spec="NaiveForecaster() * ExponentialSmoothing()")
        assert result["success"] is False
        assert "error" in result


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
