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
    _is_valid_var_name,
    export_code_tool,
)
from sktime_mcp.tools.instantiate import (
    instantiate_tool,
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


class TestVarNameValidation:
    """Tests for Python variable name validation in code export."""

    def test_valid_var_name(self):
        """A normal identifier should be accepted."""
        assert _is_valid_var_name("my_forecaster") is True

    def test_space_in_var_name_invalid(self):
        """Names with spaces should be rejected."""
        assert _is_valid_var_name("my forecaster") is False

    def test_leading_digit_invalid(self):
        """Names starting with digits should be rejected."""
        assert _is_valid_var_name("123model") is False

    def test_keyword_invalid(self):
        """Python keywords should be rejected."""
        assert _is_valid_var_name("class") is False


class TestExportCodeTool:
    """Tests for the main export_code_tool function."""

    def _create_handle(self, spec="NaiveForecaster()"):
        """Helper to create an estimator handle."""
        result = instantiate_tool(spec=spec)
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
            assert "my_forecaster = craft(" in result["code"]
        finally:
            self._cleanup_handle(handle)

    @pytest.mark.parametrize("var_name", ["my model", "123model", "class"])
    def test_invalid_var_name_fails(self, var_name):
        """Invalid Python identifiers should fail fast."""
        handle = self._create_handle()
        try:
            result = export_code_tool(handle, var_name=var_name)
            assert result["success"] is False
            assert (
                result["error"] == "var_name must be a valid Python identifier and not a keyword."
            )
        finally:
            self._cleanup_handle(handle)

    def test_export_with_params(self):
        """Params should appear in exported code."""
        handle = self._create_handle("NaiveForecaster(strategy='mean')")
        try:
            result = export_code_tool(handle)
            assert result["success"]
            assert "strategy='mean'" in result["code"] or 'strategy="mean"' in result["code"]
        finally:
            self._cleanup_handle(handle)

    def test_pipeline_handle(self):
        """Pipeline handle should return is_pipeline=True."""
        result = instantiate_tool(spec="Deseasonalizer() * NaiveForecaster()")
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
