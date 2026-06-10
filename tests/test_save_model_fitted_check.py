"""Tests for the fitted-check guard added to save_model_tool."""

import sys

sys.path.insert(0, "src")


class TestSaveModelFittedCheck:
    """Tests that save_model_tool rejects unfitted estimators early."""

    def test_save_unfitted_estimator_returns_error(self):
        """Saving an unfitted handle must fail with a clear error message."""
        from sktime_mcp.runtime.handles import get_handle_manager
        from sktime_mcp.tools.save_model import save_model_tool

        handle_manager = get_handle_manager()
        handle = handle_manager.create_handle("NaiveForecaster", object())

        try:
            result = save_model_tool(handle, "/tmp/should_not_be_created")
        finally:
            handle_manager.release_handle(handle)

        assert not result["success"]
        assert "not been fitted" in result["error"]
        assert "NaiveForecaster" in result["error"]

    def test_save_unknown_handle_still_returns_error(self):
        """Passing a nonexistent handle must still return a clear error."""
        from sktime_mcp.tools.save_model import save_model_tool

        result = save_model_tool("est_nonexistent_handle", "/tmp/model")

        assert not result["success"]
        assert "Handle not found" in result["error"]

    def test_save_fitted_estimator_proceeds(self, monkeypatch, tmp_path):
        """A fitted handle must pass the guard and reach the save call."""
        import sktime_mcp.tools.save_model as save_model_module
        from sktime_mcp.runtime.handles import get_handle_manager
        from sktime_mcp.tools.save_model import save_model_tool

        calls = {}

        def fake_save_model(**kwargs):
            calls.update(kwargs)

        monkeypatch.setattr(save_model_module, "_get_mlflow_save_model", lambda: fake_save_model)

        handle_manager = get_handle_manager()
        handle = handle_manager.create_handle("NaiveForecaster", object())
        handle_manager.mark_fitted(handle)

        try:
            result = save_model_tool(handle, str(tmp_path / "model_dir"))
        finally:
            handle_manager.release_handle(handle)

        assert result["success"], result
        assert result["saved_path"] == str(tmp_path / "model_dir")
        assert "path" in calls
