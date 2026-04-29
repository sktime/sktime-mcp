"""Tests for multi-instance creation via instantiate_estimator n_instances param."""

from sktime_mcp.tools.instantiate import instantiate_estimator_tool


class TestNInstances:
    """Test suite for the n_instances parameter of instantiate_estimator_tool."""

    def test_default_single_instance(self):
        """Default n_instances=1 should return a single handle (backward compat)."""
        result = instantiate_estimator_tool("NaiveForecaster")
        assert result["success"] is True
        assert "handle" in result
        assert "handles" not in result

    def test_explicit_single_instance(self):
        """Explicit n_instances=1 should behave the same as default."""
        result = instantiate_estimator_tool("NaiveForecaster", n_instances=1)
        assert result["success"] is True
        assert "handle" in result

    def test_multiple_instances(self):
        """n_instances=3 should return 3 independent handles."""
        result = instantiate_estimator_tool("NaiveForecaster", n_instances=3)
        assert result["success"] is True
        assert "handles" in result
        assert len(result["handles"]) == 3
        assert result["n_instances"] == 3
        # All handles should be unique
        assert len(set(result["handles"])) == 3

    def test_multiple_instances_with_params(self):
        """Params should be carried to all instances."""
        result = instantiate_estimator_tool(
            "NaiveForecaster", params={"strategy": "mean"}, n_instances=2
        )
        assert result["success"] is True
        assert len(result["handles"]) == 2
        assert result["params"] == {"strategy": "mean"}

    def test_instances_are_independent(self):
        """Fitting one instance should not affect another."""
        from sktime_mcp.runtime.handles import get_handle_manager
        from sktime_mcp.tools.fit_predict import fit_predict_tool

        result = instantiate_estimator_tool("NaiveForecaster", n_instances=2)
        assert result["success"]
        h1, h2 = result["handles"]

        hm = get_handle_manager()

        # Fit only the first
        fp = fit_predict_tool(h1, "airline", horizon=3)
        assert fp["success"]

        # First should be fitted, second should not
        assert hm.is_fitted(h1) is True
        assert hm.is_fitted(h2) is False

    def test_n_instances_zero_returns_error(self):
        """n_instances=0 should fail."""
        result = instantiate_estimator_tool("NaiveForecaster", n_instances=0)
        assert result["success"] is False
        assert "n_instances" in result["error"]

    def test_n_instances_negative_returns_error(self):
        """Negative n_instances should fail."""
        result = instantiate_estimator_tool("NaiveForecaster", n_instances=-1)
        assert result["success"] is False
        assert "n_instances" in result["error"]

    def test_n_instances_non_int_returns_error(self):
        """Non-integer n_instances should fail."""
        result = instantiate_estimator_tool("NaiveForecaster", n_instances=2.5)
        assert result["success"] is False
        assert "n_instances" in result["error"]

    def test_invalid_estimator_with_n_instances(self):
        """Unknown estimator should fail even with n_instances > 1."""
        result = instantiate_estimator_tool(
            "NonExistentEstimator", n_instances=3
        )
        assert result["success"] is False
