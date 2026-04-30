"""Tests for the diagnose_residuals_tool."""

from sktime_mcp.tools.diagnose import diagnose_residuals_tool
from sktime_mcp.tools.fit_predict import fit_predict_tool
from sktime_mcp.tools.instantiate import instantiate_estimator_tool


def test_diagnose_residuals_tool_success():
    """Test successful residual diagnosis of a fitted NaiveForecaster."""
    inst_result = instantiate_estimator_tool("sktime.forecasting.naive.NaiveForecaster")
    assert inst_result["success"] is True
    handle_id = inst_result["handle"]

    fit_result = fit_predict_tool(
        estimator_handle=handle_id,
        dataset="airline",
        horizon=1,
    )
    assert fit_result["success"] is True

    diag_result = diagnose_residuals_tool(
        estimator_handle=handle_id,
        dataset="airline",
    )

    assert diag_result["success"] is True
    assert "diagnostics" in diag_result
    assert "bias" in diag_result["diagnostics"]
    assert "autocorrelation" in diag_result["diagnostics"]
    assert "llm_hint" in diag_result
    assert isinstance(diag_result["llm_hint"], str)
    assert len(diag_result["llm_hint"]) > 0


def test_diagnose_not_fitted():
    """Test that diagnosis returns a clear error on an unfitted estimator."""
    inst_result = instantiate_estimator_tool("sktime.forecasting.naive.NaiveForecaster")
    assert inst_result["success"] is True
    handle_id = inst_result["handle"]

    diag_result = diagnose_residuals_tool(
        estimator_handle=handle_id,
        dataset="airline",
    )

    assert diag_result["success"] is False
    assert "not fitted" in diag_result["error"].lower()


def test_diagnose_invalid_handle():
    """Test that diagnosis returns a clear error for a non-existent handle."""
    diag_result = diagnose_residuals_tool(
        estimator_handle="invalid_handle_xyz",
        dataset="airline",
    )

    assert diag_result["success"] is False
    assert "not found" in diag_result["error"].lower()
