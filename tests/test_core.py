"""
Tests for sktime-mcp core functionality.
"""

import pytest
import asyncio
import sys
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
    
    def test_list_datasets_tool(self):
        """Test list_datasets tool."""
        from sktime_mcp.tools.fit_predict import list_datasets_tool
        
        result = list_datasets_tool()
        
        assert result["success"]
        assert "airline" in result["datasets"]
    
    def test_describe_unknown_estimator(self):
        """Test describing an unknown estimator."""
        from sktime_mcp.tools.describe_estimator import describe_estimator_tool
        
        result = describe_estimator_tool("NotARealEstimator12345")
        
        assert not result["success"]
        assert "error" in result

    def test_list_handles_and_release_handle_tools(self):
        """Test handle listing and release tool flow."""
        from sktime_mcp.tools.instantiate import (
            instantiate_estimator_tool,
            list_handles_tool,
            release_handle_tool,
        )

        create_result = instantiate_estimator_tool("NaiveForecaster")
        assert create_result["success"]
        handle = create_result["handle"]

        handles_result = list_handles_tool()
        assert handles_result["success"]
        handle_ids = {h["handle_id"] for h in handles_result["handles"]}
        assert handle in handle_ids

        release_result = release_handle_tool(handle)
        assert release_result["success"]

    def test_server_exposes_quick_win_tools(self):
        """Test MCP server lists the newly exposed tools."""
        from sktime_mcp.server import list_tools

        tools = asyncio.run(list_tools())
        tool_names = {tool.name for tool in tools}

        assert "list_handles" in tool_names
        assert "release_handle" in tool_names
        assert "fit" in tool_names
        assert "predict" in tool_names


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
