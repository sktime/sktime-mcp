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


class TestDataHandleLimits:
    """Tests for data handle accumulation limits (#191)."""

    def _make_executor(self, max_handles: int = 5):
        from sktime_mcp.runtime.executor import Executor

        ex = Executor()
        ex._max_data_handles = max_handles
        return ex

    def _dummy_data(self):
        return {"y": object(), "X": None, "metadata": {}, "validation": {}, "config": {}}

    def test_handles_evicted_when_limit_reached(self):
        """Oldest handles are evicted once _max_data_handles is exceeded."""
        ex = self._make_executor(max_handles=5)

        handles = []
        for i in range(6):
            hid = f"data_{i:08x}"
            ex._register_data_handle(hid, self._dummy_data())
            handles.append(hid)

        # Limit is 5; registering the 6th should have evicted at least 1 oldest
        assert len(ex._data_handles) <= 5
        # The very first handle should have been evicted
        assert handles[0] not in ex._data_handles

    def test_recent_handles_survive_eviction(self):
        """Handles added after eviction are retained."""
        ex = self._make_executor(max_handles=5)

        for i in range(6):
            hid = f"data_{i:08x}"
            ex._register_data_handle(hid, self._dummy_data())

        # The most recently added handle must still be accessible
        assert f"data_{5:08x}" in ex._data_handles

    def test_format_releases_original_handle(self):
        """format_data_handle releases the source handle after creating the formatted copy."""
        import pandas as pd

        ex = self._make_executor(max_handles=50)

        # Create a minimal data handle manually
        idx = pd.date_range("2020-01-01", periods=10, freq="D")
        y = pd.Series(range(10), index=idx, name="value")
        original_id = "data_orig0001"
        ex._data_handles[original_id] = {
            "y": y,
            "X": None,
            "metadata": {"rows": 10},
            "validation": {},
            "config": {},
        }

        result = ex.format_data_handle(
            original_id,
            auto_infer_freq=True,
            fill_missing=False,
            remove_duplicates=False,
        )

        assert result["success"]
        new_id = result["data_handle"]
        assert new_id != original_id
        # Original must be gone
        assert original_id not in ex._data_handles
        # Formatted handle must exist
        assert new_id in ex._data_handles

    def test_max_data_handles_default(self):
        """Default _max_data_handles is 50."""
        from sktime_mcp.runtime.executor import Executor

        ex = Executor()
        assert ex._max_data_handles == 50

    def test_cleanup_oldest_removes_correct_count(self):
        """_cleanup_oldest_data removes the expected number of oldest handles."""
        ex = self._make_executor(max_handles=50)

        for i in range(10):
            ex._data_handles[f"data_{i:08x}"] = self._dummy_data()

        ex._cleanup_oldest_data(count=3)

        assert len(ex._data_handles) == 7
        # First 3 (oldest) should be gone
        for i in range(3):
            assert f"data_{i:08x}" not in ex._data_handles
        # Rest survive
        for i in range(3, 10):
            assert f"data_{i:08x}" in ex._data_handles


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
