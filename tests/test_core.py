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


class TestTuningTools:
    """Tests for hyperparameter tuning tools."""

    def _make_data_handle(self, executor):
        """Create an airline data handle via the public load_data_source API."""
        from sktime.datasets import load_airline

        y = load_airline()
        result = executor.load_data_source(
            {
                "type": "pandas",
                "data": y.to_frame(),
                "target_column": y.name,
            }
        )
        assert result["success"], f"Failed to create data handle: {result.get('error')}"
        return result["data_handle"]

    def test_get_param_grid_suggestions_known_estimator(self):
        """get_param_grid_suggestions returns a dict for a known estimator."""
        from sktime_mcp.tools.tuning_tools import get_param_grid_suggestions_tool

        result = get_param_grid_suggestions_tool("NaiveForecaster")

        assert result["success"]
        assert result["estimator"] == "NaiveForecaster"
        assert isinstance(result["suggested_param_grid"], dict)

    def test_get_param_grid_suggestions_unknown_estimator(self):
        """get_param_grid_suggestions returns an error for an unknown estimator."""
        from sktime_mcp.tools.tuning_tools import get_param_grid_suggestions_tool

        result = get_param_grid_suggestions_tool("NotARealEstimator99999")

        assert not result["success"]
        assert "error" in result

    def test_tune_forecaster_grid_search(self):
        """tune_forecaster with grid search returns best_params and a new handle."""
        from sktime_mcp.runtime.executor import Executor

        executor = Executor()
        h = executor.instantiate("NaiveForecaster", {})
        assert h["success"]

        data_handle = self._make_data_handle(executor)

        result = executor.tune_forecaster(
            estimator_handle=h["handle"],
            data_handle=data_handle,
            param_grid={"strategy": ["mean", "last", "drift"]},
            method="grid",
            fh=12,
        )

        assert result["success"]
        assert "best_params" in result
        assert result["best_params"]["strategy"] in ["mean", "last", "drift"]
        assert isinstance(result["best_score"], float)
        assert result["new_handle"].startswith("est_")

    def test_tune_forecaster_random_search(self):
        """tune_forecaster with random search returns best_params and a new handle."""
        from sktime_mcp.runtime.executor import Executor

        executor = Executor()
        h = executor.instantiate("NaiveForecaster", {})
        data_handle = self._make_data_handle(executor)

        result = executor.tune_forecaster(
            estimator_handle=h["handle"],
            data_handle=data_handle,
            param_grid={"strategy": ["mean", "last", "drift"]},
            method="random",
            fh=12,
            n_iter=2,
        )

        assert result["success"]
        assert "best_params" in result
        assert result["new_handle"].startswith("est_")

    def test_tune_forecaster_invalid_estimator_handle(self):
        """tune_forecaster returns an error for a non-existent estimator handle."""
        from sktime_mcp.runtime.executor import Executor

        executor = Executor()
        data_handle = self._make_data_handle(executor)

        result = executor.tune_forecaster(
            estimator_handle="est_doesnotexist",
            data_handle=data_handle,
            param_grid={"strategy": ["mean"]},
        )

        assert not result["success"]
        assert "error" in result

    def test_tune_forecaster_invalid_data_handle(self):
        """tune_forecaster returns an error for a non-existent data handle."""
        from sktime_mcp.runtime.executor import Executor

        executor = Executor()
        h = executor.instantiate("NaiveForecaster", {})

        result = executor.tune_forecaster(
            estimator_handle=h["handle"],
            data_handle="data_doesnotexist",
            param_grid={"strategy": ["mean"]},
        )

        assert not result["success"]
        assert "error" in result

    def test_tune_forecaster_valid_scoring_metric(self):
        """tune_forecaster accepts a valid metric name and runs successfully."""
        from sktime_mcp.runtime.executor import Executor

        executor = Executor()
        h = executor.instantiate("NaiveForecaster", {})
        data_handle = self._make_data_handle(executor)

        result = executor.tune_forecaster(
            estimator_handle=h["handle"],
            data_handle=data_handle,
            param_grid={"strategy": ["mean", "last"]},
            method="grid",
            fh=12,
            scoring="MeanAbsolutePercentageError",
        )

        assert result["success"]
        assert "best_params" in result

    def test_tune_forecaster_invalid_scoring_metric(self):
        """tune_forecaster returns a structured error for an unknown metric name."""
        from sktime_mcp.runtime.executor import Executor

        executor = Executor()
        h = executor.instantiate("NaiveForecaster", {})
        data_handle = self._make_data_handle(executor)

        result = executor.tune_forecaster(
            estimator_handle=h["handle"],
            data_handle=data_handle,
            param_grid={"strategy": ["mean"]},
            method="grid",
            fh=12,
            scoring="NotARealMetric",
        )

        assert not result["success"]
        assert "NotARealMetric" in result["error"]

    def test_tune_forecaster_invalid_method(self):
        """tune_forecaster returns an error for an unsupported search method."""
        from sktime_mcp.runtime.executor import Executor

        executor = Executor()
        h = executor.instantiate("NaiveForecaster", {})
        data_handle = self._make_data_handle(executor)

        result = executor.tune_forecaster(
            estimator_handle=h["handle"],
            data_handle=data_handle,
            param_grid={"strategy": ["mean"]},
            method="unsupported_method",
        )

        assert not result["success"]
        assert "error" in result

    def test_tune_forecaster_optuna(self):
        """tune_forecaster with optuna runs if optuna is installed, errors clearly if not."""
        import importlib

        from sktime_mcp.runtime.executor import Executor

        executor = Executor()
        h = executor.instantiate("NaiveForecaster", {})
        data_handle = self._make_data_handle(executor)

        result = executor.tune_forecaster(
            estimator_handle=h["handle"],
            data_handle=data_handle,
            param_grid={"strategy": ["mean", "last"]},
            method="optuna",
            fh=12,
        )

        if importlib.util.find_spec("optuna") is None:
            assert not result["success"]
            assert "optuna" in result["error"].lower()
        else:
            assert result["success"]
            assert "best_params" in result

    def test_tune_forecaster_best_handle_is_fitted(self):
        """The new handle returned by tune_forecaster is marked as fitted."""
        from sktime_mcp.runtime.executor import Executor
        from sktime_mcp.runtime.handles import get_handle_manager

        executor = Executor()
        h = executor.instantiate("NaiveForecaster", {})
        data_handle = self._make_data_handle(executor)

        result = executor.tune_forecaster(
            estimator_handle=h["handle"],
            data_handle=data_handle,
            param_grid={"strategy": ["mean", "last"]},
            method="grid",
            fh=12,
        )

        assert result["success"]
        handle_manager = get_handle_manager()
        assert handle_manager.is_fitted(result["new_handle"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
