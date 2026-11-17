"""load_dataset must use one X/y convention for every dataset family.

Previously it returned {"data", "exog"} whose meaning flipped between
classification datasets (data=X-panel, exog=y-labels) and forecasting
datasets (data=y-target, exog=X-features). fit/predict/update assumed the
classification convention, so fit(y_dataset="longley", X_dataset="longley")
silently fitted with y and X swapped — the exogenous regressors became the
target.
"""

import pandas as pd
import pytest

from sktime_mcp.runtime.executor import get_executor
from sktime_mcp.tools.fit_predict import fit_tool
from sktime_mcp.tools.instantiate import instantiate_tool


def _release(handle):
    import contextlib

    from sktime_mcp.runtime.handles import get_handle_manager

    with contextlib.suppress(KeyError):
        get_handle_manager().release_handle(handle)


class TestLoadDatasetCanonicalKeys:
    def test_forecasting_series_dataset(self):
        res = get_executor().load_dataset("airline")
        assert res["success"]
        assert isinstance(res["y"], pd.Series)
        assert res["X"] is None

    def test_forecasting_dataset_with_exog(self):
        res = get_executor().load_dataset("longley")
        assert res["success"]
        # y is the univariate target (TOTEMP), X the exogenous regressors
        assert res["y"].ndim == 1 or res["y"].shape[1] == 1
        assert isinstance(res["X"], pd.DataFrame)
        assert res["X"].shape[1] == 5

    def test_classification_dataset(self):
        res = get_executor().load_dataset("arrow_head")
        assert res["success"]
        # X is the panel DataFrame, y the class labels — one label per instance
        assert isinstance(res["X"], pd.DataFrame)
        assert res["y"].ndim == 1
        assert len(res["X"]) == len(res["y"])


class TestFitResolution:
    def test_same_dataset_fit_uses_target_as_y(self):
        """fit(y_dataset="longley", X_dataset="longley") must fit on TOTEMP."""
        result = instantiate_tool(spec="NaiveForecaster()")
        assert result["success"]
        handle = result["handle"]
        try:
            fit_res = fit_tool(
                estimator_handle=handle,
                y_dataset="longley",
                X_dataset="longley",
            )
            assert fit_res["success"], fit_res
            instance = get_executor()._handle_manager.get_instance(handle)
            fitted_y = instance._y
            # The buggy path fitted on the 5-column exogenous frame
            assert fitted_y.ndim == 1 or fitted_y.shape[1] == 1, (
                f"y was swapped with X: fitted on shape {fitted_y.shape}"
            )
        finally:
            _release(handle)

    def test_same_dataset_fit_classifier_still_works(self):
        """Classification datasets must keep panel→X, labels→y routing."""
        result = instantiate_tool(spec="KNeighborsTimeSeriesClassifier()")
        if not result["success"]:
            pytest.skip("classifier not available")
        handle = result["handle"]
        try:
            fit_res = fit_tool(
                estimator_handle=handle,
                y_dataset="arrow_head",
                X_dataset="arrow_head",
            )
            assert fit_res["success"], fit_res
        finally:
            _release(handle)
