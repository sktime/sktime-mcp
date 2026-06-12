"""Tests for craft-based instantiation tools."""

import sys

import pytest

sys.path.insert(0, "src")

from sktime_mcp.tools.craft_utils import SpecValidationError, safe_craft, validate_spec_ast
from sktime_mcp.tools.instantiate import (
    instantiate_estimator_tool,
    instantiate_pipeline_tool,
    instantiate_tool,
    release_handle_tool,
    set_params_tool,
)


class TestCraftSecurity:
    """AST validation must block arbitrary code execution."""

    @pytest.mark.parametrize(
        "spec",
        [
            "__import__('os').system('echo pwned')",
            "open('/etc/passwd')",
            "(lambda: NaiveForecaster())()",
            "[NaiveForecaster() for _ in range(1)]",
        ],
    )
    def test_malicious_specs_rejected(self, spec):
        with pytest.raises(SpecValidationError):
            validate_spec_ast(spec)

    def test_getattr_blocked(self):
        with pytest.raises(SpecValidationError):
            validate_spec_ast("getattr(NaiveForecaster, '__init__')()")


class TestInstantiateTool:
    """Tests for the unified instantiate craft tool."""

    def test_simple_expression(self):
        result = instantiate_tool('NaiveForecaster(strategy="last")')
        assert result["success"], result
        assert result["estimator"] == "NaiveForecaster"
        assert "handle" in result
        assert "spec" in result
        assert "NaiveForecaster" in result["spec"]
        release_handle_tool(result["handle"])

    def test_dry_run_returns_deps_and_imports(self):
        result = instantiate_tool('NaiveForecaster(strategy="last")', dry_run=True)
        assert result["success"], result
        assert result["valid"] is True
        assert "handle" not in result
        assert isinstance(result["deps"], list)
        assert "NaiveForecaster" in result["imports"]

    def test_multiline_pipeline_spec(self):
        spec = """
deseason = Deseasonalizer(sp=12)
fc = NaiveForecaster(strategy='last')
return TransformedTargetForecaster(steps=[('d', deseason), ('f', fc)])
""".strip()
        result = instantiate_tool(spec)
        assert result["success"], result
        assert result["estimator"] == "TransformedTargetForecaster"
        release_handle_tool(result["handle"])

    def test_invalid_class_rejected(self):
        result = instantiate_tool("NotARealEstimator()")
        assert result["success"] is False
        assert "error" in result


class TestSetParamsTool:
    """Tests for set_params on craft-backed handles."""

    def test_set_params_updates_spec(self):
        created = instantiate_tool("NaiveForecaster()")
        assert created["success"], created
        handle = created["handle"]

        updated = set_params_tool(handle, {"strategy": "mean"})
        assert updated["success"], updated
        assert "mean" in updated["spec"]

        release_handle_tool(handle)

    def test_set_params_rejects_fitted_handle(self):
        from sktime_mcp.runtime.handles import get_handle_manager

        created = instantiate_tool("NaiveForecaster()")
        handle = created["handle"]
        get_handle_manager().mark_fitted(handle)

        result = set_params_tool(handle, {"strategy": "mean"})
        assert result["success"] is False
        assert "fitted" in result["error"].lower()

        release_handle_tool(handle)


class TestLegacyCompat:
    """Backward-compatible instantiate_estimator/pipeline wrappers."""

    def test_instantiate_estimator_wrapper(self):
        result = instantiate_estimator_tool("NaiveForecaster", {"strategy": "last"})
        assert result["success"], result
        assert "spec" in result
        release_handle_tool(result["handle"])

    def test_instantiate_pipeline_wrapper(self):
        result = instantiate_pipeline_tool(["Deseasonalizer", "NaiveForecaster"])
        assert result["success"], result
        assert result["estimator"] == "TransformedTargetForecaster"
        release_handle_tool(result["handle"])

    def test_safe_craft_round_trip(self):
        obj = safe_craft('NaiveForecaster(strategy="last")')
        round_trip = safe_craft(str(obj))
        assert type(round_trip).__name__ == "NaiveForecaster"
