"""
Tests for the code generation tool (codegen.py).

Covers Issue #69: [ENH] Add unit tests for export_code tool.
"""

import contextlib
import sys

import pytest

sys.path.insert(0, "src")

from sktime_mcp.tools.codegen import (
    _format_value,
    _generate_pipeline_code,
    _generate_single_estimator_code,
    export_code_tool,
)
from sktime_mcp.tools.instantiate import (
    instantiate_estimator_tool,
    instantiate_pipeline_tool,
)


class TestFormatValue:
    """Tests for the _format_value helper."""

    def test_string_value(self):
        """Strings should be wrapped in double quotes."""
        assert _format_value("hello") == '"hello"'

    def test_integer_value(self):
        """Integers should be returned as-is."""
        assert _format_value(42) == "42"

    def test_float_value(self):
        """Floats should be returned as-is."""
        assert _format_value(3.14) == "3.14"

    def test_bool_true(self):
        """True should render as 'True'."""
        assert _format_value(True) == "True"

    def test_bool_false(self):
        """False should render as 'False'."""
        assert _format_value(False) == "False"

    def test_none_value(self):
        """None should render as 'None'."""
        assert _format_value(None) == "None"

    def test_list_value(self):
        """Lists should use square brackets."""
        result = _format_value([1, 2, 3])
        assert result == "[1, 2, 3]"

    def test_tuple_value(self):
        """Tuples should use parentheses."""
        result = _format_value((1, 2))
        assert result == "(1, 2)"

    def test_single_element_tuple(self):
        """Single-element tuples should have trailing comma."""
        result = _format_value((1,))
        assert result == "(1,)"

    def test_dict_value(self):
        """Dicts should use curly braces with quoted keys."""
        result = _format_value({"a": 1})
        assert result == '{"a": 1}'

    def test_nested_list(self):
        """Nested lists should be formatted recursively."""
        result = _format_value([[1, 2], [3, 4]])
        assert result == "[[1, 2], [3, 4]]"


class TestSingleEstimatorCodeGen:
    """Tests for _generate_single_estimator_code."""

    def test_generates_correct_import(self):
        """Should produce the right import path for NaiveForecaster."""
        result = _generate_single_estimator_code("NaiveForecaster", {})
        assert result["success"]
        assert "from sktime" in result["code"]
        assert "import NaiveForecaster" in result["code"]

    def test_instantiation_with_params(self):
        """Params should appear in the generated code."""
        result = _generate_single_estimator_code("NaiveForecaster", {"strategy": "last"})
        assert result["success"]
        assert 'strategy="last"' in result["code"]

    def test_instantiation_without_params(self):
        """Empty params should produce empty parentheses."""
        result = _generate_single_estimator_code("NaiveForecaster", {})
        assert result["success"]
        assert "NaiveForecaster()" in result["code"]

    def test_custom_var_name(self):
        """Custom var_name should be used in the output."""
        result = _generate_single_estimator_code("NaiveForecaster", {}, var_name="forecaster")
        assert result["success"]
        assert "forecaster = NaiveForecaster()" in result["code"]

    def test_unknown_estimator_fails(self):
        """Unknown estimator should return success=False."""
        result = _generate_single_estimator_code("NotARealEstimator99999", {})
        assert not result["success"]
        assert "error" in result

    def test_composite_params_include_step_imports(self):
        """Composite estimator params should include imports for nested step classes."""
        from sktime.forecasting.naive import NaiveForecaster
        from sktime.transformations.series.difference import Differencer

        result = _generate_single_estimator_code(
            "ForecastingPipeline",
            {
                "steps": [
                    ("differencer", Differencer()),
                    ("forecaster", NaiveForecaster()),
                ]
            },
            var_name="pipeline",
        )

        assert result["success"]
        assert "import ForecastingPipeline" in result["code"]
        assert "import Differencer" in result["code"]
        assert "import NaiveForecaster" in result["code"]


class TestPipelineCodeGen:
    """Tests for _generate_pipeline_code."""

    def test_transformer_forecaster_pipeline(self):
        """Transformer + Forecaster should use TransformedTargetForecaster."""
        result = _generate_pipeline_code(
            ["Deseasonalizer", "NaiveForecaster"],
            [{}, {}],
        )
        assert result["success"]
        assert "TransformedTargetForecaster" in result["code"]

    def test_all_transformers_pipeline(self):
        """All transformers should use TransformerPipeline."""
        result = _generate_pipeline_code(
            ["Deseasonalizer", "Detrender"],
            [{}, {}],
        )
        assert result["success"]
        assert "TransformerPipeline" in result["code"]

    def test_unknown_component_fails(self):
        """Unknown component in pipeline should fail."""
        result = _generate_pipeline_code(
            ["NotRealEstimator12345", "NaiveForecaster"],
            [{}, {}],
        )
        assert not result["success"]
        assert "error" in result

    def test_step_ordering(self):
        """Steps should be numbered correctly (step_0, step_1)."""
        result = _generate_pipeline_code(
            ["Deseasonalizer", "NaiveForecaster"],
            [{}, {}],
        )
        assert result["success"]
        assert "step_0" in result["code"]
        assert "step_1" in result["code"]

    def test_pipeline_with_params(self):
        """Pipeline components should include their params."""
        result = _generate_pipeline_code(
            ["NaiveForecaster"],
            [{"strategy": "mean"}],
        )
        assert result["success"]
        assert 'strategy="mean"' in result["code"]


class TestExportCodeTool:
    """Tests for the main export_code_tool function."""

    def _create_handle(self, name="NaiveForecaster", params=None):
        """Helper to create an estimator handle."""
        result = instantiate_estimator_tool(name, params)
        assert result["success"], f"Failed to create handle: {result}"
        return result["handle"]

    def _cleanup_handle(self, handle):
        """Helper to release a handle."""
        from sktime_mcp.runtime.handles import get_handle_manager

        with contextlib.suppress(KeyError):
            get_handle_manager().release_handle(handle)

    def test_export_success(self):
        """Valid handle should return success with code."""
        handle = self._create_handle()
        try:
            result = export_code_tool(handle)
            assert result["success"]
            assert "code" in result
            assert "estimator_name" in result
            assert "is_pipeline" in result
            assert "handle" in result
            assert result["estimator_name"] == "NaiveForecaster"
            assert result["is_pipeline"] is False
        finally:
            self._cleanup_handle(handle)

    def test_unknown_handle_fails(self):
        """Non-existent handle should return success=False."""
        result = export_code_tool("est_nonexistent_handle_xyz")
        assert not result["success"]
        assert "error" in result
        assert "not found" in result["error"].lower()

    def test_include_fit_example_true(self):
        """With include_fit_example=True, code should contain fit/predict."""
        handle = self._create_handle()
        try:
            result = export_code_tool(handle, include_fit_example=True)
            assert result["success"]
            assert "load_airline" in result["code"]
            assert ".fit(" in result["code"]
            assert ".predict(" in result["code"]
        finally:
            self._cleanup_handle(handle)

    def test_include_fit_example_false(self):
        """With include_fit_example=False, code should NOT contain fit example."""
        handle = self._create_handle()
        try:
            result = export_code_tool(handle, include_fit_example=False)
            assert result["success"]
            assert "load_airline" not in result["code"]
        finally:
            self._cleanup_handle(handle)

    def test_custom_var_name(self):
        """Custom var_name should appear in the generated code."""
        handle = self._create_handle()
        try:
            result = export_code_tool(handle, var_name="my_forecaster")
            assert result["success"]
            assert "my_forecaster" in result["code"]
        finally:
            self._cleanup_handle(handle)

    def test_export_with_params(self):
        """Params should appear in exported code."""
        handle = self._create_handle("NaiveForecaster", {"strategy": "mean"})
        try:
            result = export_code_tool(handle)
            assert result["success"]
            assert 'strategy="mean"' in result["code"]
        finally:
            self._cleanup_handle(handle)

    def test_pipeline_handle(self):
        """Pipeline handle should return is_pipeline=True."""
        result = instantiate_pipeline_tool(["Deseasonalizer", "NaiveForecaster"])
        if not result["success"]:
            pytest.skip("Pipeline instantiation not available")

        handle = result["handle"]
        try:
            export_result = export_code_tool(handle)
            assert export_result["success"]
            assert export_result["is_pipeline"] is True
        finally:
            self._cleanup_handle(handle)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
