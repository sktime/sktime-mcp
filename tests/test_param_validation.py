"""
Tests for parameter validation in instantiate_estimator tool.

Covers Issue #3: [ENH] Add type validation for params in instantiate_estimator.
"""

import sys
import types

import pytest

sys.path.insert(0, "src")

from sktime_mcp.tools.fit_predict import predict_tool
from sktime_mcp.tools.instantiate import (
    _validate_params,
    instantiate_estimator_tool,
    instantiate_pipeline_tool,
    list_handles_tool,
    load_model_tool,
    release_handle_tool,
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

    def test_non_string_key_rejected(self):
        """Top-level non-string keys should be rejected."""
        result = _validate_params({1: "value"})
        assert result["valid"] is False
        assert "Parameter keys must be strings" in result["error"]

    def test_nested_dict_non_string_key_rejected(self):
        """Nested dicts with non-string keys should be rejected as unsafe."""
        result = _validate_params({"config": {1: "bad-key"}})
        assert result["valid"] is False
        assert "Unsupported type" in result["error"]

    def test_unknown_key_returns_error(self):
        """Unknown param key should fail validation with clear valid-key list."""
        result = _validate_params(
            {"nonexistent_param_xyz": 1},
            estimator_name="NaiveForecaster",
        )
        assert result["valid"] is False
        assert "Unknown parameter(s)" in result["error"]
        assert "nonexistent_param_xyz" in result["error"]
        assert "Valid parameters" in result["error"]


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

    def test_unknown_key_returns_error(self):
        """Unknown param key should return success=False with clear guidance."""
        result = instantiate_estimator_tool(
            "NaiveForecaster",
            {"nonexistent_param_xyz": 1},
        )
        assert result["success"] is False
        assert "Unknown parameter(s)" in result["error"]
        assert "nonexistent_param_xyz" in result["error"]

    def test_attaches_validation_warnings_when_successful(self, monkeypatch):
        """Estimator tool should attach warnings when validation returns them."""

        class DummyExecutor:
            def instantiate(self, estimator, params):
                return {"success": True, "handle": "est_123", "estimator": estimator}

        monkeypatch.setattr(
            "sktime_mcp.tools.instantiate._validate_params",
            lambda params, estimator_name=None: {"valid": True, "warnings": ["warn-a"]},
        )
        monkeypatch.setattr("sktime_mcp.tools.instantiate.get_executor", lambda: DummyExecutor())

        result = instantiate_estimator_tool("NaiveForecaster", {"strategy": "last"})
        assert result["success"] is True
        assert result["warnings"] == ["warn-a"]


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

    def test_pipeline_success_attaches_aggregated_warnings(self, monkeypatch):
        """Pipeline tool should aggregate warnings from component validations."""
        validation_results = iter(
            [
                {"valid": True, "warnings": ["w0"]},
                {"valid": True, "warnings": ["w1"]},
            ]
        )

        class DummyExecutor:
            def instantiate_pipeline(self, components, params_list):
                return {"success": True, "handle": "pipe_123", "components": components}

        monkeypatch.setattr(
            "sktime_mcp.tools.instantiate._validate_params",
            lambda params, estimator_name=None: next(validation_results),
        )
        monkeypatch.setattr("sktime_mcp.tools.instantiate.get_executor", lambda: DummyExecutor())

        result = instantiate_pipeline_tool(
            ["CompA", "CompB"],
            [{"a": 1}, {"b": 2}],
        )
        assert result["success"] is True
        assert result["warnings"] == ["w0", "w1"]


class TestHandleTools:
    """Tests for handle-management helper tools."""

    def test_release_handle_messages(self, monkeypatch):
        """release_handle_tool should map bool result to human-friendly message."""

        class DummyHandleManager:
            def release_handle(self, handle):
                return handle == "exists"

        monkeypatch.setattr(
            "sktime_mcp.tools.instantiate.get_handle_manager",
            lambda: DummyHandleManager(),
        )

        ok = release_handle_tool("exists")
        missing = release_handle_tool("missing")
        assert ok["success"] is True
        assert ok["message"] == "Handle released"
        assert missing["success"] is False
        assert missing["message"] == "Handle not found"

    def test_list_handles_returns_count(self, monkeypatch):
        """list_handles_tool should return all handles with count."""
        handles = [{"handle": "h1"}, {"handle": "h2"}]

        class DummyHandleManager:
            def list_handles(self):
                return handles

        monkeypatch.setattr(
            "sktime_mcp.tools.instantiate.get_handle_manager",
            lambda: DummyHandleManager(),
        )
        result = list_handles_tool()
        assert result["success"] is True
        assert result["handles"] == handles
        assert result["count"] == 2


class TestLoadModelTool:
    """Tests for load_model_tool import, success, and failure paths."""

    def test_load_model_missing_mlflow_dependency(self, monkeypatch):
        """If mlflow helper import fails, function returns a clear dependency error."""
        monkeypatch.delitem(sys.modules, "sktime.utils.mlflow_sktime", raising=False)
        result = load_model_tool("/tmp/model-path")
        assert result["success"] is False
        assert "mlflow" in result["error"].lower()

    def test_load_model_success(self, monkeypatch):
        """Successful load should create + fit handle and return metadata."""

        class DummyEstimator:
            pass

        fake_module = types.ModuleType("sktime.utils.mlflow_sktime")
        fake_module.load_model = lambda path: DummyEstimator()
        monkeypatch.setitem(sys.modules, "sktime.utils.mlflow_sktime", fake_module)

        class DummyHandleManager:
            def __init__(self):
                self.marked = None

            def create_handle(self, estimator_name, instance, params, metadata):
                assert estimator_name == "DummyEstimator"
                assert params == {}
                assert metadata["source"] == "loaded"
                return "est_loaded_1"

            def mark_fitted(self, handle):
                self.marked = handle

        hm = DummyHandleManager()
        monkeypatch.setattr("sktime_mcp.tools.instantiate.get_handle_manager", lambda: hm)

        result = load_model_tool("/tmp/model-path")
        assert result["success"] is True
        assert result["handle"] == "est_loaded_1"
        assert result["estimator"] == "DummyEstimator"
        assert hm.marked == "est_loaded_1"

    def test_load_model_runtime_failure(self, monkeypatch):
        """Unexpected load exceptions should be surfaced as tool errors."""
        fake_module = types.ModuleType("sktime.utils.mlflow_sktime")
        fake_module.load_model = lambda path: (_ for _ in ()).throw(RuntimeError("boom"))
        monkeypatch.setitem(sys.modules, "sktime.utils.mlflow_sktime", fake_module)

        result = load_model_tool("/tmp/model-path")
        assert result["success"] is False
        assert "Failed to load model" in result["error"]
        assert result["path"] == "/tmp/model-path"


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
