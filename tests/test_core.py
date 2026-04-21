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


class TestTransformTools:
    """Tests for fit_transform and transform tools (Issue #167)."""

    def test_fit_transform_demo_dataset(self):
        """fit_transform on a demo dataset returns transformed series."""
        from sktime_mcp.tools.instantiate import instantiate_estimator_tool
        from sktime_mcp.tools.transform_tools import fit_transform_tool

        inst = instantiate_estimator_tool("Detrender")
        assert inst["success"], inst.get("error")
        handle = inst["handle"]

        result = fit_transform_tool(handle, dataset="airline")

        assert result["success"], result.get("error")
        assert "transformed" in result
        assert result["n_timepoints"] > 0
        assert result["output_type"] in ("series", "dataframe")
        assert result["estimator_handle"] == handle

    def test_fit_transform_marks_handle_fitted(self):
        """fit_transform should mark the estimator handle as fitted."""
        from sktime_mcp.runtime.handles import get_handle_manager
        from sktime_mcp.tools.instantiate import instantiate_estimator_tool
        from sktime_mcp.tools.transform_tools import fit_transform_tool

        inst = instantiate_estimator_tool("Detrender")
        handle = inst["handle"]

        assert not get_handle_manager().is_fitted(handle)
        fit_transform_tool(handle, dataset="airline")
        assert get_handle_manager().is_fitted(handle)

    def test_transform_after_fit_transform(self):
        """transform should succeed after fit_transform has been called."""
        from sktime_mcp.tools.instantiate import instantiate_estimator_tool
        from sktime_mcp.tools.transform_tools import fit_transform_tool, transform_tool

        inst = instantiate_estimator_tool("Detrender")
        handle = inst["handle"]

        fit_result = fit_transform_tool(handle, dataset="airline")
        assert fit_result["success"]

        transform_result = transform_tool(handle, dataset="airline")
        assert transform_result["success"], transform_result.get("error")
        assert "transformed" in transform_result
        assert transform_result["n_timepoints"] > 0

    def test_transform_without_fit_fails(self):
        """transform on an unfitted handle should return an error."""
        from sktime_mcp.tools.instantiate import instantiate_estimator_tool
        from sktime_mcp.tools.transform_tools import transform_tool

        inst = instantiate_estimator_tool("Detrender")
        handle = inst["handle"]

        result = transform_tool(handle, dataset="airline")

        assert not result["success"]
        assert "not fitted" in result["error"].lower()

    def test_fit_transform_no_data_source_errors(self):
        """fit_transform without dataset or data_handle should return an error."""
        from sktime_mcp.tools.instantiate import instantiate_estimator_tool
        from sktime_mcp.tools.transform_tools import fit_transform_tool

        inst = instantiate_estimator_tool("Detrender")
        result = fit_transform_tool(inst["handle"])

        assert not result["success"]
        assert "error" in result

    def test_fit_transform_unknown_handle_errors(self):
        """fit_transform with a non-existent handle should return an error."""
        from sktime_mcp.tools.transform_tools import fit_transform_tool

        result = fit_transform_tool("est_doesnotexist", dataset="airline")

        assert not result["success"]
        assert "not found" in result["error"].lower()

    def test_server_exposes_transform_tools(self):
        """list_tools should include fit_transform and transform."""
        import asyncio

        from sktime_mcp.server import list_tools

        tools = asyncio.run(list_tools())
        names = {t.name for t in tools}
        assert "fit_transform" in names
        assert "transform" in names


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
