"""
analyze_data tool for sktime MCP.

Provides statistical analysis of time series data including
stationarity, trend, and seasonality detection.
"""

import logging
from typing import Any, Optional

import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import acf, adfuller, pacf

from sktime_mcp.runtime.executor import get_executor

logger = logging.getLogger(__name__)


def _compute_basic_stats(y: pd.Series) -> dict[str, Any]:
    """Compute basic time series statistics."""
    stats = {
        "length": len(y),
        "is_nan": float(y.isna().sum()),
        "is_zero": float((y == 0).sum()),
        "min": float(y.min()) if not y.isna().all() else None,
        "max": float(y.max()) if not y.isna().all() else None,
        "mean": float(y.mean()) if not y.isna().all() else None,
        "std": float(y.std()) if not y.isna().all() else None,
    }

    # Frequency detection
    freq = y.index.freq
    if freq is None:
        freq = pd.infer_freq(y.index)

    stats["frequency"] = str(freq) if freq else None
    stats["has_index"] = True

    # Time range
    if len(y) > 0:
        stats["start"] = str(y.index.min())
        stats["end"] = str(y.index.max())

    return stats


def _compute_stationarity(y: pd.Series) -> dict[str, Any]:
    """Compute stationarity using Augmented Dickey-Fuller test."""
    clean_y = y.dropna()

    if len(clean_y) < 10:
        return {
            "test": "adf",
            "statistic": None,
            "p_value": None,
            "critical_values": None,
            "is_stationary": None,
            "error": "Insufficient data for ADF test (need at least 10 points)",
        }

    try:
        result = adfuller(clean_y, autolag="AIC")

        return {
            "test": "adf",
            "statistic": float(result[0]),
            "p_value": float(result[1]),
            "critical_values": {k: float(v) for k, v in result[4].items()},
            "is_stationary": bool(result[1] < 0.05),
            "lags_used": int(result[2]),
        }
    except Exception as e:
        logger.warning(f"ADF test failed: {e}")
        return {
            "test": "adf",
            "statistic": None,
            "p_value": None,
            "critical_values": None,
            "is_stationary": None,
            "error": str(e),
        }


def _compute_trend(y: pd.Series) -> dict[str, Any]:
    """Compute trend using linear regression."""
    clean_y = y.dropna()

    if len(clean_y) < 3:
        return {
            "has_trend": None,
            "trend_slope": None,
            "trend_intercept": None,
            "trend_p_value": None,
            "r_squared": None,
            "error": "Insufficient data for trend analysis (need at least 3 points)",
        }

    try:
        # Use index as x values (numeric)
        x = np.arange(len(clean_y))
        y_values = clean_y.values

        # Linear regression using numpy polyfit
        slope, intercept = np.polyfit(x, y_values, 1)

        # Calculate R-squared
        y_pred = slope * x + intercept
        ss_res = np.sum((y_values - y_pred) ** 2)
        ss_tot = np.sum((y_values - np.mean(y_values)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

        # Compute p-value using statsmodels for proper significance
        from scipy import stats as scipy_stats

        n = len(x)
        df = n - 2  # degrees of freedom
        se = np.sqrt(ss_res / df)
        se_slope = se / np.sqrt(np.sum((x - np.mean(x)) ** 2))
        t_stat = slope / se_slope if se_slope != 0 else 0
        p_value = 2 * (1 - scipy_stats.t.cdf(abs(t_stat), df))

        # Determine if trend is significant
        has_trend = bool(abs(p_value) < 0.05 and abs(slope) > 1e-6)

        return {
            "has_trend": has_trend,
            "trend_slope": float(slope),
            "trend_intercept": float(intercept),
            "trend_p_value": float(p_value),
            "r_squared": float(r_squared),
        }
    except Exception as e:
        logger.warning(f"Trend analysis failed: {e}")
        return {
            "has_trend": None,
            "trend_slope": None,
            "trend_intercept": None,
            "trend_p_value": None,
            "r_squared": None,
            "error": str(e),
        }


def _compute_seasonality(y: pd.Series, max_lags: int = 24) -> dict[str, Any]:
    """Compute seasonality using ACF analysis."""
    clean_y = y.dropna()

    if len(clean_y) < 10:
        return {
            "has_seasonality": None,
            "seasonal_strength": None,
            "seasonal_period": None,
            "acf_values": None,
            "pacf_values": None,
            "ljung_box_p_value": None,
            "error": "Insufficient data for seasonality analysis (need at least 10 points)",
        }

    try:
        # Determine appropriate number of lags
        n_lags = min(max_lags, len(clean_y) // 2 - 1)

        if n_lags < 2:
            return {
                "has_seasonality": None,
                "seasonal_strength": None,
                "seasonal_period": None,
                "acf_values": None,
                "pacf_values": None,
                "ljung_box_p_value": None,
                "error": "Insufficient data for seasonality analysis",
            }

        # Compute ACF and PACF
        acf_values = acf(clean_y, nlags=n_lags, fft=True)
        pacf_values = pacf(clean_y, nlags=n_lags)

        # Ljung-Box test for autocorrelation
        try:
            lb_result = acorr_ljungbox(clean_y, lags=[min(10, n_lags)], return_df=True)
            lb_p_value = float(lb_result["lb_pvalue"].iloc[0])
        except Exception:
            lb_p_value = None

        # Detect dominant seasonal period by finding peaks in ACF
        seasonal_period = None
        seasonal_strength = None

        if len(acf_values) > 2:
            # Look for peaks beyond lag 0
            acf_no_lag0 = acf_values[1:]
            if len(acf_no_lag0) > 0:
                # Find indices where ACF is above confidence interval
                conf_int = 1.96 / np.sqrt(len(clean_y))
                significant_lags = np.where(acf_no_lag0 > conf_int)[0]

                if len(significant_lags) > 0:
                    # Find the first significant lag after lag 0
                    seasonal_period = int(significant_lags[0] + 1)
                    seasonal_strength = float(acf_values[seasonal_period]) if seasonal_period < len(acf_values) else None

        # Determine if seasonality is present
        # Based on significant autocorrelation at seasonal lags
        has_seasonality = seasonal_period is not None and seasonal_strength is not None and seasonal_strength > conf_int

        return {
            "has_seasonality": bool(has_seasonality) if has_seasonality is not None else None,
            "seasonal_strength": float(seasonal_strength) if seasonal_strength is not None else None,
            "seasonal_period": seasonal_period,
            "acf_values": [float(v) for v in acf_values],
            "pacf_values": [float(v) for v in pacf_values],
            "ljung_box_p_value": lb_p_value,
            "confidence_interval": float(1.96 / np.sqrt(len(clean_y))),
        }
    except Exception as e:
        logger.warning(f"Seasonality analysis failed: {e}")
        return {
            "has_seasonality": None,
            "seasonal_strength": None,
            "seasonal_period": None,
            "acf_values": None,
            "pacf_values": None,
            "ljung_box_p_value": None,
            "error": str(e),
        }


def _compute_summary(
    stationarity: dict[str, Any],
    trend: dict[str, Any],
    seasonality: dict[str, Any],
) -> dict[str, Any]:
    """Compute a summary interpretation of the analysis."""
    summary_parts = []

    # Stationarity summary
    if stationarity.get("is_stationary") is True:
        summary_parts.append("Series is stationary")
    elif stationarity.get("is_stationary") is False:
        summary_parts.append("Series is non-stationary (has unit root)")
    elif stationarity.get("is_stationary") is None:
        summary_parts.append(f"Stationarity unknown ({stationarity.get('error', 'test failed')})")

    # Trend summary
    if trend.get("has_trend") is True:
        slope = trend.get("trend_slope", 0)
        direction = "increasing" if slope > 0 else "decreasing"
        summary_parts.append(f"Series has a significant {direction} trend (slope={slope:.4f})")
    elif trend.get("has_trend") is False:
        summary_parts.append("No significant trend detected")
    elif trend.get("has_trend") is None:
        summary_parts.append(f"Trend analysis inconclusive ({trend.get('error', 'test failed')})")

    # Seasonality summary
    if seasonality.get("has_seasonality") is True:
        period = seasonality.get("seasonal_period")
        strength = seasonality.get("seasonal_strength", 0)
        summary_parts.append(f"Seasonality detected (period={period}, strength={strength:.3f})")
    elif seasonality.get("has_seasonality") is False:
        summary_parts.append("No significant seasonality detected")
    elif seasonality.get("has_seasonality") is None:
        summary_parts.append(f"Seasonality unknown ({seasonality.get('error', 'test failed')})")

    return {
        "interpretation": "; ".join(summary_parts),
        "recommendations": _get_recommendations(stationarity, trend, seasonality),
    }


def _get_recommendations(
    stationarity: dict[str, Any],
    trend: dict[str, Any],
    seasonality: dict[str, Any],
) -> list[str]:
    """Get modeling recommendations based on analysis."""
    recommendations = []

    if stationarity.get("is_stationary") is False:
        recommendations.append("Consider differencing or Detrender to make series stationary")
        recommendations.append("ARIMA with d>0 or SARIMA may be appropriate")

    if trend.get("has_trend") is True:
        recommendations.append("Consider including trend component in model")
        recommendations.append("Detrender transformer may help")

    if seasonality.get("has_seasonality") is True:
        period = seasonality.get("seasonal_period")
        if period:
            recommendations.append(f"Consider seasonal models (period={period})")
            recommendations.append("SARIMA, Prophet, or seasonal exponential smoothing recommended")

    if (
        stationarity.get("is_stationary") is not False
        and trend.get("has_trend") is not True
        and seasonality.get("has_seasonality") is not True
    ):
        recommendations.append("Simple models like ARIMA or exponential smoothing may suffice")

    if not recommendations:
        recommendations.append("Further analysis needed before selecting a model")

    return recommendations


def analyze_data_tool(
    data_handle: str,
    include_acf: bool = True,
    max_acf_lags: int = 24,
) -> dict[str, Any]:
    """
    Analyze a loaded time series dataset and return statistical characteristics.

    Computes stationarity (ADF test), trend (linear regression), and
    seasonality (ACF analysis) to help guide model selection.

    Args:
        data_handle: Handle from load_data_source
        include_acf: Whether to include full ACF/PACF values (default: True)
        max_acf_lags: Maximum number of lags for ACF analysis (default: 24)

    Returns:
        Dictionary with:
        - success: bool
        - data_handle: str
        - basic_stats: dict (length, frequency, min, max, mean, std, etc.)
        - stationarity: dict (ADF test results)
        - trend: dict (linear regression results)
        - seasonality: dict (ACF analysis results)
        - summary: dict (interpretation and recommendations)
        - errors: list of any errors encountered during analysis

    Example:
        >>> analyze_data_tool(data_handle="data_abc123")
        {
            "success": True,
            "data_handle": "data_abc123",
            "basic_stats": {"length": 144, "frequency": "MS", ...},
            "stationarity": {"test": "adf", "p_value": 0.01, "is_stationary": True, ...},
            "trend": {"has_trend": False, "trend_slope": 0.001, ...},
            "seasonality": {"has_seasonality": True, "seasonal_period": 12, ...},
            "summary": {"interpretation": "...", "recommendations": [...]}
        }
    """
    executor = get_executor()

    # Validate data handle exists
    if data_handle not in executor._data_handles:
        return {
            "success": False,
            "error": f"Unknown data handle: {data_handle}",
            "available_handles": list(executor._data_handles.keys()),
        }

    data_info = executor._data_handles[data_handle]
    y = data_info["y"]
    errors = []

    # Validate input
    if y is None or (isinstance(y, pd.Series) and len(y) == 0):
        return {
            "success": False,
            "error": "Data series is empty",
        }

    # Check if data is all NaN
    if isinstance(y, pd.Series) and y.isna().all():
        return {
            "success": False,
            "error": "Data series contains only NaN values",
        }

    try:
        # Compute basic statistics
        basic_stats = _compute_basic_stats(y)

        # Compute stationarity
        stationarity = _compute_stationarity(y)

        # Compute trend
        trend = _compute_trend(y)

        # Compute seasonality
        seasonality = _compute_seasonality(y, max_lags=max_acf_lags)

        # Optionally truncate ACF values for cleaner output
        if not include_acf:
            seasonality["acf_values"] = None
            seasonality["pacf_values"] = None

        # Compute summary
        summary = _compute_summary(stationarity, trend, seasonality)

        return {
            "success": True,
            "data_handle": data_handle,
            "basic_stats": basic_stats,
            "stationarity": stationarity,
            "trend": trend,
            "seasonality": seasonality,
            "summary": summary,
        }

    except Exception as e:
        logger.exception("Error during data analysis")
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__,
        }
