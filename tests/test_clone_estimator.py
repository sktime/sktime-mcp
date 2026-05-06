"""Tests for the clone_estimator MCP tool."""

from sktime_mcp.tools.instantiate import (
    clone_estimator_tool,
    instantiate_estimator_tool,
)


class TestCloneEstimatorTool:
    """Test suite for clone_estimator_tool."""

    def test_clone_unfitted_estimator(self):
        """Cloning an unfitted estimator should produce a new unfitted handle."""
        inst = instantiate_estimator_tool("NaiveForecaster")
        assert inst["success"]
        original_handle = inst["handle"]

        result = clone_estimator_tool(original_handle)
        assert result["success"] is True
        assert result["original_handle"] == original_handle
        assert result["cloned_handle"] != original_handle
        assert result["estimator"] == "NaiveForecaster"
        assert "cloned_handle" in result

    def test_clone_fitted_estimator_produces_unfitted_copy(self):
        """Cloning a fitted estimator should give an unfitted clone."""
        from sktime_mcp.runtime.handles import get_handle_manager
        from sktime_mcp.tools.fit_predict import fit_predict_tool

        inst = instantiate_estimator_tool("NaiveForecaster")
        assert inst["success"]
        handle = inst["handle"]

        fp = fit_predict_tool(handle, "airline", horizon=3)
        assert fp["success"]

        hm = get_handle_manager()
        assert hm.is_fitted(handle) is True

        result = clone_estimator_tool(handle)
        assert result["success"] is True
        cloned_handle = result["cloned_handle"]

        # The clone should be unfitted
        assert hm.is_fitted(cloned_handle) is False

    def test_cloned_estimator_is_independent(self):
        """Fitting the clone should not affect the original."""
        from sktime_mcp.runtime.handles import get_handle_manager
        from sktime_mcp.tools.fit_predict import fit_predict_tool

        inst = instantiate_estimator_tool("NaiveForecaster")
        assert inst["success"]
        handle = inst["handle"]

        result = clone_estimator_tool(handle)
        assert result["success"]
        cloned_handle = result["cloned_handle"]

        hm = get_handle_manager()

        # Fit only the clone
        fp = fit_predict_tool(cloned_handle, "airline", horizon=3)
        assert fp["success"]

        # Original should remain unfitted
        assert hm.is_fitted(handle) is False
        assert hm.is_fitted(cloned_handle) is True

    def test_clone_with_params(self):
        """Clone should carry over hyperparameters."""
        inst = instantiate_estimator_tool(
            "NaiveForecaster", params={"strategy": "mean"}
        )
        assert inst["success"]

        result = clone_estimator_tool(inst["handle"])
        assert result["success"] is True
        assert result["params"] == {"strategy": "mean"}

    def test_clone_nonexistent_handle(self):
        """Cloning a non-existent handle should fail gracefully."""
        result = clone_estimator_tool("est_nonexistent")
        assert result["success"] is False
        assert "Handle not found" in result["error"]

    def test_clone_empty_handle(self):
        """Empty handle string should return a clear error."""
        result = clone_estimator_tool("")
        assert result["success"] is False
        assert "non-empty string" in result["error"]

    def test_clone_non_string_handle(self):
        """Non-string handle should return a clear error."""
        result = clone_estimator_tool(42)
        assert result["success"] is False
        assert "non-empty string" in result["error"]

    def test_two_clones_have_different_handles(self):
        """Two clones of the same estimator should have different handles."""
        inst = instantiate_estimator_tool("NaiveForecaster")
        assert inst["success"]
        handle = inst["handle"]

        clone1 = clone_estimator_tool(handle)
        clone2 = clone_estimator_tool(handle)

        assert clone1["success"] and clone2["success"]
        assert clone1["cloned_handle"] != clone2["cloned_handle"]
