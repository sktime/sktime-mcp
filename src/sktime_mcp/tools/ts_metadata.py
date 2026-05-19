"""
extract_ts_metadata tool for sktime MCP.

Provides rich statistical profiling of time series data for
LLM agents to reason about data characteristics before model selection.
"""

import logging
from typing import Any, Optional

import numpy as np
import pandas as pd

from sktime_mcp.runtime.executor import get_executor

logger = logging.getLogger(__name__)


def _detect_trend(y: pd.Series) -> str:
    """Detect trend direction using linear regression slope."""
    try:
        x = np.arange(len(y))
        slope = np.polyfit(x, y.values.astype(float), 1)[0]
        # Normalize slope relative to mean to determine significance
        relative_slope = abs(slope) / (abs(float(y.mean())) + 1e-10)
        if relative_slope < 0.001:
            return "none"
        return "upward" if slope > 0 else "downward"
    except Exception:
        return "unknown"


def _check_stationarity(y: pd.Series) -> dict[str, Any]:
    """Run Augmented Dickey-Fuller test for stationarity."""
    try:
        from statsmodels.tsa.stattools import adfuller

        result = adfuller(y.dropna(), autolag="AIC")
        adf_stat = float(result[0])
        p_value = float(result[1])
        return {
            "adf_statistic": round(adf_stat, 4),
            "adf_pvalue": round(p_value, 4),
            "is_stationary": bool(p_value < 0.05),
        }
    except Exception as e:
        logger.debug(f"Stationarity test failed: {e}")
        return {"error": str(e), "is_stationary": None}


def _detect_seasonality(y: pd.Series, freq: Optional[str]) -> dict[str, Any]:
    """Detect seasonality using ACF analysis."""
    try:
        from statsmodels.tsa.stattools import acf

        n = len(y)
        if n < 10:
            return {"detected": False, "dominant_period": None, "strength": "none"}

        # Map known frequencies to their natural seasonal period
        freq_period_map = {
            "MS": [12],
            "M": [12],
            "ME": [12],
            "QS": [4],
            "Q": [4],
            "QE": [4],
            "D": [7, 365],
            "h": [24, 168],
            "H": [24, 168],
            "W": [52],
            "A": [1],
            "Y": [1],
            "YE": [1],
            "min": [60, 1440],
            "T": [60, 1440],
        }
        candidates = freq_period_map.get(freq, [4, 7, 12, 24, 52]) if freq else [4, 7, 12, 24, 52]

        max_lag = min(n // 2 - 1, max(candidates) + 5) if candidates else min(n // 2 - 1, 50)
        if max_lag < 2:
            return {"detected": False, "dominant_period": None, "strength": "none"}

        acf_values = acf(y.dropna(), nlags=max_lag, fft=True)

        # Find peak ACF at candidate periods first
        best_period = None
        best_acf = 0.0
        for period in candidates:
            if period < len(acf_values):
                val = abs(acf_values[period])
                if val > best_acf:
                    best_acf = val
                    best_period = period

        # If no strong candidate found, scan all lags for any peak
        if best_acf < 0.3:
            for lag in range(2, len(acf_values)):
                val = abs(acf_values[lag])
                if val > best_acf:
                    best_acf = val
                    best_period = lag

        detected = bool(best_acf > 0.3)
        if best_acf > 0.7:
            strength = "strong"
        elif best_acf > 0.4:
            strength = "moderate"
        elif best_acf > 0.3:
            strength = "weak"
        else:
            strength = "none"

        return {
            "detected": detected,
            "dominant_period": best_period if detected else None,
            "strength": strength,
        }
    except Exception as e:
        logger.debug(f"Seasonality detection failed: {e}")
        return {"detected": False, "dominant_period": None, "strength": "none", "error": str(e)}


def _get_autocorrelation(y: pd.Series, seasonal_period: Optional[int]) -> dict[str, Any]:
    """Compute autocorrelation at lag 1 and the seasonal lag."""
    try:
        from statsmodels.tsa.stattools import acf

        max_lag = max(2, (seasonal_period or 1) + 1)
        max_lag = min(max_lag, len(y) // 2 - 1)
        if max_lag < 1:
            return {"lag_1": None, "lag_seasonal": None}

        acf_vals = acf(y.dropna(), nlags=max_lag, fft=True)
        lag1 = round(float(acf_vals[1]), 4) if len(acf_vals) > 1 else None
        lag_s = (
            round(float(acf_vals[seasonal_period]), 4)
            if seasonal_period and seasonal_period < len(acf_vals)
            else None
        )
        return {"lag_1": lag1, "lag_seasonal": lag_s}
    except Exception as e:
        logger.debug(f"Autocorrelation calculation failed: {e}")
        return {"lag_1": None, "lag_seasonal": None}


def _recommend_models(
    is_stationary: Optional[bool],
    seasonality: dict,
    trend: str,
) -> list[str]:
    """
    Suggest estimators from the sktime registry based on data characteristics.
    Returns up to 5 model names that are actually present in the registry.
    """
    try:
        from sktime_mcp.registry.interface import get_registry

        registry = get_registry()
        all_forecasters = registry.get_all_estimators(task="forecasting")
        registry_names = {node.name for node in all_forecasters}

        # Priority-ordered candidates per characteristic
        if seasonality.get("detected"):
            candidates = [
                "AutoARIMA",
                "ExponentialSmoothing",
                "TBATS",
                "Prophet",
                "ARIMA",
                "AutoETS",
            ]
        elif is_stationary is False:
            candidates = [
                "ARIMA",
                "AutoARIMA",
                "ExponentialSmoothing",
                "ThetaForecaster",
                "AutoETS",
            ]
        else:
            candidates = [
                "NaiveForecaster",
                "ExponentialSmoothing",
                "ARIMA",
                "AutoARIMA",
                "ThetaForecaster",
            ]

        recommended = [m for m in candidates if m in registry_names]

        # Always include NaiveForecaster as a baseline if not already present
        if "NaiveForecaster" not in recommended and "NaiveForecaster" in registry_names:
            recommended.append("NaiveForecaster")

        return recommended[:5]
    except Exception as e:
        logger.debug(f"Model recommendation failed: {e}")
        return ["NaiveForecaster", "ARIMA", "ExponentialSmoothing"]


def _build_agent_hints(
    trend: str,
    stationarity: dict,
    seasonality: dict,
    missing_pct: float,
) -> list[str]:
    """Generate actionable natural-language hints for LLM agents."""
    hints = []

    if missing_pct > 0:
        hints.append(
            f"{missing_pct:.1f}% missing values detected — "
            "run format_time_series to fill gaps before fitting"
        )

    if trend in ("upward", "downward"):
        hints.append(
            f"{trend.capitalize()} trend detected — "
            "consider adding a Detrender transformer in a pipeline"
        )

    if seasonality.get("detected"):
        period = seasonality.get("dominant_period")
        strength = seasonality.get("strength", "")
        hints.append(
            f"{strength.capitalize()} seasonality at period {period} — "
            f"use ConditionalDeseasonalizer or seasonal ARIMA (s={period})"
        )

    is_stationary = stationarity.get("is_stationary")
    p_value = stationarity.get("adf_pvalue")
    if is_stationary is False:
        hints.append(
            f"Non-stationary series (ADF p-value={p_value}) — "
            "ARIMA with d≥1, ExponentialSmoothing, or AutoARIMA recommended"
        )
    elif is_stationary is True:
        hints.append(
            f"Stationary series (ADF p-value={p_value}) — "
            "wide model choice; ARIMA with d=0 or ThetaForecaster work well"
        )

    if not hints:
        hints.append("No strong patterns detected — NaiveForecaster is a solid baseline")

    return hints


def extract_ts_metadata_tool(
    data_handle: Optional[str] = None,
    dataset: Optional[str] = None,
) -> dict[str, Any]:
    """
    Extract rich statistical metadata from a time series.

    Enables LLM agents to reason about data characteristics (trend,
    stationarity, seasonality, autocorrelation) before choosing a model —
    closing the gap in the prompt → model → evaluation agentic loop.

    Args:
        data_handle: Handle from load_data_source (custom data)
        dataset: Demo dataset name (e.g. 'airline', 'sunspots')

    Returns:
        Dictionary with statistical profile, stationarity test results,
        seasonality detection, autocorrelation, recommended models,
        and actionable agent hints.
    """
    if data_handle is None and dataset is None:
        return {"success": False, "error": "Provide either 'data_handle' or 'dataset'"}

    executor = get_executor()

    # --- Load the series ---
    if data_handle is not None:
        if data_handle not in executor._data_handles:
            return {
                "success": False,
                "error": f"Unknown data handle: {data_handle}",
                "available_handles": list(executor._data_handles.keys()),
            }
        data_info = executor._data_handles[data_handle]
        y = data_info["y"]
        freq_str = (
            str(y.index.freq)
            if getattr(y.index, "freq", None)
            else (data_info.get("metadata", {}).get("frequency"))
        )
    else:
        data_result = executor.load_dataset(dataset)
        if not data_result["success"]:
            return data_result
        y = data_result["data"]
        freq_str = (
            str(y.index.freq) if hasattr(y, "index") and getattr(y.index, "freq", None) else None
        )

    if not isinstance(y, pd.Series):
        return {
            "success": False,
            "error": "extract_ts_metadata supports univariate pd.Series only",
        }

    y_clean = y.dropna()
    n = len(y)
    missing_count = int(y.isna().sum())
    missing_pct = round(missing_count / n * 100, 2) if n > 0 else 0.0

    # --- Basic descriptive stats ---
    basic = {
        "series_length": n,
        "frequency": freq_str,
        "start_date": str(y.index.min()),
        "end_date": str(y.index.max()),
        "missing_count": missing_count,
        "missing_pct": missing_pct,
        "mean": round(float(y_clean.mean()), 4) if len(y_clean) > 0 else None,
        "std": round(float(y_clean.std()), 4) if len(y_clean) > 0 else None,
        "min": round(float(y_clean.min()), 4) if len(y_clean) > 0 else None,
        "max": round(float(y_clean.max()), 4) if len(y_clean) > 0 else None,
    }

    too_short = len(y_clean) < 10

    # --- Trend ---
    trend = _detect_trend(y_clean) if not too_short else "unknown"

    # --- Stationarity ---
    stationarity = (
        _check_stationarity(y_clean)
        if not too_short
        else {"is_stationary": None, "note": "Too few observations for ADF test"}
    )

    # --- Seasonality ---
    seasonality = (
        _detect_seasonality(y_clean, freq_str)
        if not too_short
        else {"detected": False, "dominant_period": None, "strength": "none"}
    )

    # --- Autocorrelation ---
    autocorrelation = _get_autocorrelation(y_clean, seasonality.get("dominant_period"))

    # --- Recommended models ---
    recommended_models = _recommend_models(
        stationarity.get("is_stationary"),
        seasonality,
        trend,
    )

    # --- Agent hints ---
    agent_hints = _build_agent_hints(trend, stationarity, seasonality, missing_pct)

    return {
        "success": True,
        **basic,
        "trend": trend,
        "stationarity": stationarity,
        "seasonality": seasonality,
        "autocorrelation": autocorrelation,
        "recommended_models": recommended_models,
        "agent_hints": agent_hints,
    }
