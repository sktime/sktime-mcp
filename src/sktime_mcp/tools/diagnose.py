"""
diagnose tool for sktime MCP.

Executes post-fit residual diagnostics on a fitted estimator and
translates statistical test results into actionable LLM hints.
"""

import logging
from typing import Any

import numpy as np
from scipy.stats import shapiro
from statsmodels.stats.diagnostic import acorr_ljungbox

from sktime_mcp.runtime.executor import get_executor

logger = logging.getLogger(__name__)


def diagnose_residuals_tool(
    estimator_handle: str,
    dataset: str,
    significance_level: float = 0.05,
) -> dict[str, Any]:
    """
    Diagnose residuals of a fitted estimator.

    Computes in-sample residuals and runs statistical tests (Ljung-Box,
    Shapiro-Wilk, bias) to give an LLM actionable reasoning about why
    the model failed and what to try next.

    Args:
        estimator_handle: Handle ID from instantiate_estimator
        dataset: Name of demo dataset (e.g. 'airline', 'sunspots')
        significance_level: Alpha threshold for statistical tests (default 0.05)

    Returns:
        Dictionary with success status, diagnostics dict, and llm_hint string
    """
    executor = get_executor()

    # --- 1. Retrieve the fitted instance ---
    try:
        instance = executor._handle_manager.get_instance(estimator_handle)
    except KeyError:
        return {"success": False, "error": f"Handle not found: {estimator_handle}"}

    if not executor._handle_manager.is_fitted(estimator_handle):
        return {
            "success": False,
            "error": f"Estimator {estimator_handle} is not fitted. Call fit_predict first.",
        }

    # --- 2. Reload the training data (same pattern as evaluate.py) ---
    data_result = executor.load_dataset(dataset)
    if not data_result["success"]:
        return data_result

    y = data_result["data"]

    # --- 3. Compute in-sample residuals (primary + fallback) ---
    try:
        residuals = instance.predict_residuals(y=y)
    except Exception:
        try:
            preds = instance.predict(fh=y.index)
            residuals = y - preds
        except Exception as e:
            return {"success": False, "error": f"Failed to compute residuals: {e}"}

    # Sanitize: some models (ARIMA, differencing) produce leading NaNs
    residuals = residuals.dropna()
    res_array = np.asarray(residuals, dtype=float)

    if len(res_array) < 3:
        return {
            "success": False,
            "error": "Not enough residual data points to run statistical tests.",
        }

    # --- 4. Statistical tests ---
    diagnostics: dict[str, Any] = {}
    llm_hints: list[str] = []

    # Bias (mean error)
    mean_error = float(np.mean(res_array))
    std_error = float(np.std(res_array))

    if mean_error > 0.1 * std_error:
        bias_status = "consistently under-forecasting"
        llm_hints.append(
            f"Model has a persistent positive bias ({mean_error:.4f}), "
            "consistently under-forecasting the actual values. "
            "Consider adding a drift term or using an unbiased baseline."
        )
    elif mean_error < -0.1 * std_error:
        bias_status = "consistently over-forecasting"
        llm_hints.append(
            f"Model has a persistent negative bias ({mean_error:.4f}), "
            "consistently over-forecasting the actual values. "
            "Consider adding a drift term or using an unbiased baseline."
        )
    else:
        bias_status = "unbiased"

    diagnostics["bias"] = {
        "mean_error": mean_error,
        "status": bias_status,
    }

    # Autocorrelation (Ljung-Box)
    lb_lags = max(2, min(10, len(res_array) // 5))
    try:
        lb_res = acorr_ljungbox(res_array, lags=[lb_lags], return_df=True)
        lb_p_value = float(lb_res["lb_pvalue"].iloc[0])
        lb_passed = lb_p_value >= significance_level

        diagnostics["autocorrelation"] = {
            "ljung_box_passed": lb_passed,
            "p_value": lb_p_value,
            "tested_lags": lb_lags,
        }

        if not lb_passed:
            llm_hints.append(
                f"Residuals show significant autocorrelation at lag {lb_lags} "
                f"(p={lb_p_value:.4f}). The model failed to capture time dependence "
                "or seasonality. Consider switching to SARIMA or adding a Deseasonalizer pipeline."
            )
    except Exception as e:
        logger.warning(f"Ljung-Box test failed: {e}")
        diagnostics["autocorrelation"] = {"error": str(e)}

    # Normality (Shapiro-Wilk), capped at 500 samples
    shapiro_data = res_array[-500:] if len(res_array) > 500 else res_array
    try:
        _, shapiro_p_value = shapiro(shapiro_data)
        shapiro_p_value = float(shapiro_p_value)
        shapiro_passed = shapiro_p_value >= significance_level

        diagnostics["normality"] = {
            "shapiro_passed": shapiro_passed,
            "p_value": shapiro_p_value,
        }

        if not shapiro_passed:
            llm_hints.append(
                f"Residuals are not normally distributed (Shapiro p={shapiro_p_value:.4f}), "
                "suggesting heavy-tailed noise or non-linear effects not captured by the current model."
            )
    except Exception as e:
        logger.warning(f"Shapiro-Wilk test failed: {e}")
        diagnostics["normality"] = {"error": str(e)}

    # --- 5. Build final LLM hint ---
    if llm_hints:
        final_hint = " ".join(llm_hints)
    else:
        final_hint = (
            "Residuals look like white noise: no significant autocorrelation, "
            "no strong bias, and approximately normal distribution. "
            "The model has captured the underlying signal well."
        )

    return {
        "success": True,
        "diagnostics": diagnostics,
        "llm_hint": final_hint,
    }
