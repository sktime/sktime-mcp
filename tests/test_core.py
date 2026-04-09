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
        if result["estimators"]:
            estimator = result["estimators"][0]
            assert "tags" in estimator
            assert "tag_count" in estimator
            assert "hyperparameter_count" in estimator
            assert "docstring_preview" in estimator

    def test_fit_predict_classification_tool(self):
        """Classification tool should fit on supervised data and predict on feature-only data."""
        from sktime.datasets import load_basic_motions

        from sktime_mcp.tools.data_tools import load_data_source_tool, release_data_handle_tool
        from sktime_mcp.tools.instantiate import instantiate_estimator_tool, release_handle_tool
        from sktime_mcp.tools.supervised import fit_predict_classification_tool

        X_train, y_train = load_basic_motions(split="train")
        X_test, _ = load_basic_motions(split="test")

        train_df = X_train.copy()
        train_df["target"] = y_train

        train_result = load_data_source_tool(
            {
                "type": "pandas",
                "data": train_df,
                "target_column": "target",
            }
        )
        predict_result = load_data_source_tool(
            {
                "type": "pandas",
                "data": X_test.copy(),
                "feature_only": True,
            }
        )
        estimator_result = instantiate_estimator_tool(
            "DummyClassifier",
            {"strategy": "most_frequent"},
        )

        try:
            assert train_result["success"]
            assert predict_result["success"]
            assert predict_result["metadata"]["feature_only"] is True
            assert estimator_result["success"]

            result = fit_predict_classification_tool(
                estimator_result["handle"],
                train_result["data_handle"],
                predict_result["data_handle"],
                return_probabilities=True,
            )

            assert result["success"]
            assert result["task"] == "classification"
            assert len(result["predictions"]) == len(X_test)
            assert len(result["probabilities"]) == len(X_test)
            assert "classes" in result
        finally:
            if estimator_result.get("success"):
                release_handle_tool(estimator_result["handle"])
            if train_result.get("success"):
                release_data_handle_tool(train_result["data_handle"])
            if predict_result.get("success"):
                release_data_handle_tool(predict_result["data_handle"])

    def test_fit_predict_regression_tool(self):
        """Regression tool should fit on supervised data and predict numeric outputs."""
        from sktime.datasets import load_tecator

        from sktime_mcp.tools.data_tools import load_data_source_tool, release_data_handle_tool
        from sktime_mcp.tools.instantiate import instantiate_estimator_tool, release_handle_tool
        from sktime_mcp.tools.supervised import fit_predict_regression_tool

        X_train, y_train = load_tecator(split="train")
        X_test, _ = load_tecator(split="test")

        train_df = X_train.copy()
        train_df["target"] = y_train

        train_result = load_data_source_tool(
            {
                "type": "pandas",
                "data": train_df,
                "target_column": "target",
            }
        )
        predict_result = load_data_source_tool(
            {
                "type": "pandas",
                "data": X_test.copy(),
                "feature_only": True,
            }
        )
        estimator_result = instantiate_estimator_tool(
            "DummyRegressor",
            {"strategy": "mean"},
        )

        try:
            assert train_result["success"]
            assert predict_result["success"]
            assert estimator_result["success"]

            result = fit_predict_regression_tool(
                estimator_result["handle"],
                train_result["data_handle"],
                predict_result["data_handle"],
            )

            assert result["success"]
            assert result["task"] == "regression"
            assert len(result["predictions"]) == len(X_test)
            assert all(isinstance(value, (float, int)) for value in result["predictions"])
        finally:
            if estimator_result.get("success"):
                release_handle_tool(estimator_result["handle"])
            if train_result.get("success"):
                release_data_handle_tool(train_result["data_handle"])
            if predict_result.get("success"):
                release_data_handle_tool(predict_result["data_handle"])

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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
