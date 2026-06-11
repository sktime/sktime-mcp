"""Tests for dataset tracking in export_code / fit_predict interaction."""

import sys

sys.path.insert(0, "src")


class TestExportCodeDatasetTracking:
    """Tests that export_code uses the training dataset from handle metadata."""

    def test_fit_predict_stores_training_dataset_in_metadata(self):
        """After fit_predict, the handle metadata must record the dataset name."""
        from sktime_mcp.runtime.handles import get_handle_manager
        from sktime_mcp.tools.fit_predict import fit_predict_tool
        from sktime_mcp.tools.instantiate import instantiate_estimator_tool

        inst = instantiate_estimator_tool("NaiveForecaster", {"strategy": "last"})
        assert inst["success"], inst
        handle = inst["handle"]

        result = fit_predict_tool(handle, "sunspots", horizon=3)
        assert result["success"], result

        hm = get_handle_manager()
        info = hm.get_info(handle)
        assert info.metadata.get("training_dataset") == "sunspots", (
            f"Expected 'sunspots' in handle metadata, got: {info.metadata}"
        )

    def test_export_code_uses_training_dataset_by_default(self):
        """export_code include_fit_example=True must use the training dataset, not 'airline'."""
        from sktime_mcp.tools.codegen import export_code_tool
        from sktime_mcp.tools.fit_predict import fit_predict_tool
        from sktime_mcp.tools.instantiate import instantiate_estimator_tool

        inst = instantiate_estimator_tool("NaiveForecaster", {"strategy": "last"})
        assert inst["success"], inst
        handle = inst["handle"]

        fit_result = fit_predict_tool(handle, "sunspots", horizon=3)
        assert fit_result["success"], fit_result

        code_result = export_code_tool(handle, include_fit_example=True)
        assert code_result["success"], code_result

        code = code_result["code"]
        assert "sunspots" in code, f"Expected 'sunspots' to appear in exported code, got:\n{code}"

    def test_export_code_explicit_dataset_overrides_metadata(self):
        """An explicit dataset argument to export_code must take priority over metadata."""
        from sktime_mcp.tools.codegen import export_code_tool
        from sktime_mcp.tools.fit_predict import fit_predict_tool
        from sktime_mcp.tools.instantiate import instantiate_estimator_tool

        inst = instantiate_estimator_tool("NaiveForecaster", {"strategy": "last"})
        assert inst["success"], inst
        handle = inst["handle"]

        fit_result = fit_predict_tool(handle, "sunspots", horizon=3)
        assert fit_result["success"], fit_result

        # Explicitly request airline even though model was trained on sunspots
        code_result = export_code_tool(handle, include_fit_example=True, dataset="airline")
        assert code_result["success"], code_result

        code = code_result["code"]
        assert "airline" in code, f"Expected 'airline' to appear when explicitly set, got:\n{code}"

    def test_export_code_falls_back_to_airline_without_fit(self):
        """export_code on a never-fitted handle must fall back to 'airline'."""
        from sktime_mcp.tools.codegen import export_code_tool
        from sktime_mcp.tools.instantiate import instantiate_estimator_tool

        inst = instantiate_estimator_tool("NaiveForecaster", {"strategy": "last"})
        assert inst["success"], inst
        handle = inst["handle"]

        code_result = export_code_tool(handle, include_fit_example=True)
        assert code_result["success"], code_result

        code = code_result["code"]
        assert "airline" in code, f"Expected 'airline' fallback for unfitted handle, got:\n{code}"
