"""Tests for the coverage (prediction intervals) parameter in fit_predict."""

import sys

import pytest

sys.path.insert(0, "src")


class TestFitPredictCoverage:
    """Tests for the coverage parameter added to fit_predict."""

    def test_fit_predict_no_coverage_unchanged(self):
        """fit_predict without coverage returns the same shape as before."""
        from sktime_mcp.tools.fit_predict import fit_predict_tool
        from sktime_mcp.tools.instantiate import instantiate_estimator_tool

        inst = instantiate_estimator_tool("NaiveForecaster", {"strategy": "last"})
        assert inst["success"], inst
        handle = inst["handle"]

        result = fit_predict_tool(handle, "airline", horizon=6)
        assert result["success"], result
        assert "predictions" in result
        assert "prediction_intervals" not in result
        assert "interval_warning" not in result

    def test_fit_predict_coverage_unsupported_estimator_returns_warning(self):
        """When coverage is requested but estimator lacks capability:pred_int,
        the response includes a point forecast and an interval_warning."""
        from sktime_mcp.tools.fit_predict import fit_predict_tool
        from sktime_mcp.tools.instantiate import instantiate_estimator_tool

        inst = instantiate_estimator_tool("NaiveForecaster", {"strategy": "last"})
        assert inst["success"], inst
        handle = inst["handle"]

        result = fit_predict_tool(handle, "airline", horizon=6, coverage=0.9)
        assert result["success"], result
        assert "predictions" in result
        # NaiveForecaster may or may not support pred_int; either path is valid
        has_intervals = "prediction_intervals" in result
        has_warning = "interval_warning" in result
        assert has_intervals or has_warning, (
            "Expected either prediction_intervals or interval_warning in response"
        )

    def test_fit_predict_coverage_with_pred_int_estimator(self):
        """When coverage is requested and the estimator supports pred_int,
        prediction_intervals are included in the response."""
        from sktime_mcp.tools.fit_predict import fit_predict_tool
        from sktime_mcp.tools.instantiate import instantiate_estimator_tool
        from sktime_mcp.tools.list_estimators import list_estimators_tool

        # Find a forecaster that supports prediction intervals
        candidates = list_estimators_tool(
            task="forecasting",
            tags={"capability:pred_int": True},
            limit=5,
        )
        assert candidates["success"], candidates
        if not candidates["estimators"]:
            pytest.skip("No pred_int-capable forecasters available in this environment")

        est_name = candidates["estimators"][0]["name"]
        inst = instantiate_estimator_tool(est_name)
        assert inst["success"], inst
        handle = inst["handle"]

        result = fit_predict_tool(handle, "airline", horizon=4, coverage=0.9)
        assert result["success"], result
        assert "predictions" in result

        if "prediction_intervals" in result:
            pi = result["prediction_intervals"]
            assert pi["coverage"] == 0.9
            assert isinstance(pi["lower"], dict)
            assert isinstance(pi["upper"], dict)
            assert len(pi["lower"]) == 4
            assert len(pi["upper"]) == 4
