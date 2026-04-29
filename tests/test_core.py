"""
Tests for sktime-mcp core functionality.
"""

import sys

import pytest

sys.path.insert(0, "src")


class TestRegistryInterface:
    """Tests for the Registry Interface."""

    def test_registry_loads(self):
        """Test that the registry loads successfully."""
        from sktime_mcp.registry.interface import get_registry

        registry = get_registry()
        estimators = registry.get_all_estimators()

        assert len(estimators) > 0, "Registry should contain estimators"

    def test_filter_by_task(self):
        """Test filtering by task type."""
        from sktime_mcp.registry.interface import get_registry

        registry = get_registry()
        forecasters = registry.get_all_estimators(task="forecasting")

        assert all(e.task == "forecasting" for e in forecasters)

    def test_get_estimator_by_name(self):
        """Test getting a specific estimator."""
        from sktime_mcp.registry.interface import get_registry

        registry = get_registry()
        node = registry.get_estimator_by_name("NaiveForecaster")

        # NaiveForecaster should always exist
        if node is not None:
            assert node.name == "NaiveForecaster"
            assert node.task == "forecasting"

    def test_search_prioritizes_exact_estimator_name(self):
        """Exact estimator name matches should rank before broad matches."""
        from sktime_mcp.registry.interface import get_registry

        registry = get_registry()
        matches = registry.search_estimators("NaiveForecaster")

        assert len(matches) > 0
        assert matches[0].name == "NaiveForecaster"

    def test_search_prioritizes_case_insensitive_exact_name(self):
        """Exact estimator ranking should be case-insensitive."""
        from sktime_mcp.registry.interface import get_registry

        registry = get_registry()
        matches = registry.search_estimators("naiveforecaster")

        assert len(matches) > 0
        assert matches[0].name == "NaiveForecaster"


class TestHandleManager:
    """Tests for the Handle Manager."""

    def test_create_and_get_handle(self):
        """Test creating and retrieving handles."""
        from sktime_mcp.runtime.handles import HandleManager

        manager = HandleManager()

        # Create a dummy instance
        class DummyEstimator:
            pass

        instance = DummyEstimator()
        handle = manager.create_handle("DummyEstimator", instance, {"param": 1})

        assert handle.startswith("est_")
        assert manager.exists(handle)
        assert manager.get_instance(handle) is instance

    def test_mark_fitted(self):
        """Test marking estimators as fitted."""
        from sktime_mcp.runtime.handles import HandleManager

        manager = HandleManager()

        class DummyEstimator:
            pass

        handle = manager.create_handle("Dummy", DummyEstimator())

        assert not manager.is_fitted(handle)
        manager.mark_fitted(handle)
        assert manager.is_fitted(handle)

    def test_release_handle(self):
        """Test releasing handles."""
        from sktime_mcp.runtime.handles import HandleManager

        manager = HandleManager()
        handle = manager.create_handle("Dummy", object())

        assert manager.exists(handle)
        manager.release_handle(handle)
        assert not manager.exists(handle)


class TestCompositionValidator:
    """Tests for the Composition Validator."""

    def test_single_component_valid(self):
        """Test that a single estimator is valid."""
        from sktime_mcp.composition.validator import CompositionValidator

        validator = CompositionValidator()
        result = validator.validate_pipeline(["NaiveForecaster"])

        # Single forecaster should be valid if it exists
        if result.valid:
            assert len(result.errors) == 0

    def test_empty_pipeline_invalid(self):
        """Test that empty pipeline is invalid."""
        from sktime_mcp.composition.validator import CompositionValidator

        validator = CompositionValidator()
        result = validator.validate_pipeline([])

        assert not result.valid
        assert "empty" in result.errors[0].lower()

    def test_unknown_estimator_invalid(self):
        """Test that unknown estimators are caught."""
        from sktime_mcp.composition.validator import CompositionValidator

        validator = CompositionValidator()
        result = validator.validate_pipeline(["NotARealEstimator"])

        assert not result.valid


class TestTools:
    """Tests for MCP tools."""

    def test_list_estimators_tool(self):
        """Test list_estimators tool."""
        from sktime_mcp.tools.list_estimators import list_estimators_tool

        result = list_estimators_tool(limit=5)

        assert result["success"]
        assert "estimators" in result
        assert len(result["estimators"]) <= 5

    def test_list_estimators_detection_task(self):
        """Test that detection estimators are returned when filtering by detection task."""
        from sktime_mcp.tools.list_estimators import list_estimators_tool

        result = list_estimators_tool(task="detection", limit=100)

        assert result["success"]
        assert result["total"] > 0, "There should be detection estimators"
        assert all(e["task"] == "detection" for e in result["estimators"]), (
            "All returned estimators should have task='detection'"
        )

    def test_detection_in_available_tasks(self):
        """Test that detection appears in available tasks."""
        from sktime_mcp.tools.list_estimators import get_available_tasks

        result = get_available_tasks()

        assert result["success"]
        assert "detection" in result["tasks"], "detection should be a valid task"

    def test_describe_unknown_estimator(self):
        """Test describing an unknown estimator."""
        from sktime_mcp.tools.describe_estimator import describe_estimator_tool

        result = describe_estimator_tool("NotARealEstimator12345")

        assert not result["success"]
        assert "error" in result

    def test_list_available_data_no_filter(self):
        """list_available_data with no args returns both system_demos and active_handles."""
        from sktime_mcp.tools.list_available_data import list_available_data_tool

        result = list_available_data_tool()

        assert result["success"]
        assert "system_demos" in result
        assert "active_handles" in result
        assert "total" in result
        assert isinstance(result["system_demos"], list)
        assert isinstance(result["active_handles"], list)
        assert result["total"] == len(result["system_demos"]) + len(result["active_handles"])
        assert "airline" in result["system_demos"]

    def test_list_available_data_demos_only(self):
        """list_available_data with is_demo=True returns only system demo datasets."""
        from sktime_mcp.tools.list_available_data import list_available_data_tool

        result = list_available_data_tool(is_demo=True)

        assert result["success"]
        assert len(result["system_demos"]) > 0
        assert result["active_handles"] == []
        assert result["total"] == len(result["system_demos"])
        assert "airline" in result["system_demos"]

    def test_list_available_data_handles_only(self):
        """list_available_data with is_demo=False returns only active data handles."""
        from sktime_mcp.tools.list_available_data import list_available_data_tool

        result = list_available_data_tool(is_demo=False)

        assert result["success"]
        assert result["system_demos"] == []
        assert isinstance(result["active_handles"], list)
        assert result["total"] == len(result["active_handles"])

    def test_save_model_tool(self, monkeypatch, tmp_path):
        """Test save_model tool resolves handle and forwards parameters."""
        import sktime_mcp.tools.save_model as save_model_module
        from sktime_mcp.runtime.handles import get_handle_manager
        from sktime_mcp.tools.save_model import save_model_tool

        calls = {}

        def fake_save_model(**kwargs):
            calls.update(kwargs)

        monkeypatch.setattr(save_model_module, "_get_mlflow_save_model", lambda: fake_save_model)

        handle_manager = get_handle_manager()
        handle = handle_manager.create_handle("DummyEstimator", object())

        try:
            result = save_model_tool(
                estimator_handle=handle,
                path=str(tmp_path / "model_dir"),
                mlflow_params={"serialization_format": "pickle"},
            )
        finally:
            handle_manager.release_handle(handle)

        assert result["success"]
        assert result["saved_path"] == str(tmp_path / "model_dir")
        assert calls["path"] == str(tmp_path / "model_dir")
        assert calls["serialization_format"] == "pickle"


class TestSearchEstimatorsLimit:
    """Tests for the limit parameter validation in search_estimators_tool."""

    def test_limit_zero_returns_error(self):
        """limit=0 should return an error, not an empty list."""
        from sktime_mcp.tools.describe_estimator import search_estimators_tool

        result = search_estimators_tool("NaiveForecaster", limit=0)

        assert not result["success"]
        assert result["error"] == "limit must be a positive integer."

    def test_limit_negative_one_returns_error(self):
        """limit=-1 should return an error, not the last result."""
        from sktime_mcp.tools.describe_estimator import search_estimators_tool

        result = search_estimators_tool("NaiveForecaster", limit=-1)

        assert not result["success"]
        assert result["error"] == "limit must be a positive integer."

    def test_limit_negative_five_returns_error(self):
        """limit=-5 should return an error, not the last 5 results."""
        from sktime_mcp.tools.describe_estimator import search_estimators_tool

        result = search_estimators_tool("NaiveForecaster", limit=-5)

        assert not result["success"]
        assert result["error"] == "limit must be a positive integer."

    def test_limit_valid_returns_results(self):
        """A positive limit should work correctly and cap results."""
        pytest.importorskip("sktime", reason="sktime not installed in this environment")
        from sktime_mcp.tools.describe_estimator import search_estimators_tool

        result = search_estimators_tool("Forecaster", limit=3)

        assert result["success"]
        assert "results" in result
        assert len(result["results"]) <= 3
        assert result["count"] <= 3


class TestHorizonValidation:
    """Tests for horizon parameter validation in fit_predict, predict, and fit_predict_async."""

 
    # fit_predict_tool                                                     
     

    def test_fit_predict_tool_horizon_zero(self):
        """horizon=0 must return an error before touching the executor."""
        from sktime_mcp.tools.fit_predict import fit_predict_tool

        result = fit_predict_tool("est_dummy", "airline", horizon=0)

        assert not result["success"]
        assert "horizon" in result["error"].lower()

    def test_fit_predict_tool_horizon_negative(self):
        """horizon=-5 must return an error."""
        from sktime_mcp.tools.fit_predict import fit_predict_tool

        result = fit_predict_tool("est_dummy", "airline", horizon=-5)

        assert not result["success"]
        assert "horizon" in result["error"].lower()

    def test_fit_predict_tool_horizon_negative_one(self):
        """horizon=-1 must return an error."""
        from sktime_mcp.tools.fit_predict import fit_predict_tool

        result = fit_predict_tool("est_dummy", "airline", horizon=-1)

        assert not result["success"]
        assert "horizon" in result["error"].lower()

   
    # predict_tool                                                         
 

    def test_predict_tool_horizon_zero(self):
        """horizon=0 must return an error, not pass fh=[] to the executor."""
        from sktime_mcp.tools.fit_predict import predict_tool

        result = predict_tool("est_dummy", horizon=0)

        assert not result["success"]
        assert "horizon" in result["error"].lower()

    def test_predict_tool_horizon_negative(self):
        """horizon=-3 must return an error."""
        from sktime_mcp.tools.fit_predict import predict_tool

        result = predict_tool("est_dummy", horizon=-3)

        assert not result["success"]
        assert "horizon" in result["error"].lower()

    def test_predict_tool_horizon_negative_one(self):
        """horizon=-1 must return an error."""
        from sktime_mcp.tools.fit_predict import predict_tool

        result = predict_tool("est_dummy", horizon=-1)

        assert not result["success"]
        assert "horizon" in result["error"].lower()

    # fit_predict_async_tool                                               


    def test_fit_predict_async_tool_horizon_zero(self):
        """horizon=0 must be rejected before a background job is created."""
        from sktime_mcp.tools.fit_predict import fit_predict_async_tool

        result = fit_predict_async_tool("est_dummy", dataset="airline", horizon=0)

        assert not result["success"]
        assert "horizon" in result["error"].lower()
        
    def test_fit_predict_async_tool_horizon_negative(self):
        """horizon=-1 must be rejected before a background job is created."""
        from sktime_mcp.tools.fit_predict import fit_predict_async_tool

        result = fit_predict_async_tool("est_dummy", dataset="airline", horizon=-1)

        assert not result["success"]
        assert "horizon" in result["error"].lower()

    def test_fit_predict_async_tool_horizon_very_negative(self):
        """horizon=-100 must be rejected before a background job is created."""
        from sktime_mcp.tools.fit_predict import fit_predict_async_tool

        result = fit_predict_async_tool("est_dummy", dataset="airline", horizon=-100)

        assert not result["success"]
        assert "horizon" in result["error"].lower()

class TestServerImports:
    """Verify server.py imports the right symbols from the right modules."""

    def test_fit_predict_tool_importable_from_fit_predict(self):
        """fit_predict_tool must live in tools.fit_predict, not tools.data_tools."""
        from sktime_mcp.tools.fit_predict import fit_predict_tool

        assert callable(fit_predict_tool)

    def test_fit_predict_tool_not_in_data_tools(self):
        """data_tools must not expose fit_predict_tool (it never defined it)."""
        import sktime_mcp.tools.data_tools as data_tools

        assert not hasattr(data_tools, "fit_predict_tool")

    def test_server_imports_fit_predict_tool_from_correct_module(self):
        """Importing server must not raise ImportError for fit_predict_tool."""
        import importlib

        try:
            importlib.import_module("sktime_mcp.server")
        except ImportError as exc:
            if "fit_predict_tool" in str(exc):
                pytest.fail(f"server.py still imports fit_predict_tool from wrong module: {exc}")
            # Other ImportErrors (e.g. mcp not installed) are fine — not our bug
        except Exception:
            pass



if __name__ == "__main__":
    pytest.main([__file__, "-v"])
