"""
Dataset statistical analysis tool for sktime MCP.

Provides tools for querying statistical characteristics of a loaded time series.
"""

import logging
from typing import Any

from sktime_mcp.runtime.executor import get_executor

logger = logging.getLogger(__name__)


def analyze_data_tool(data_handle: str) -> dict[str, Any]:
    """
    Analyze standard statistical characteristics of a time series.

    Args:
        data_handle: Handle from load_data_source

    Returns:
        Dictionary with statistical summary including length, frequency,
        stationarity, trend, and seasonality.
    """
    executor = get_executor()

    if data_handle not in executor._data_handles:
        return {"success": False, "error": f"Unknown data handle: {data_handle}"}

    data_info = executor._data_handles[data_handle]
    y = data_info["y"]

    import numpy as np
    import pandas as pd
    from scipy.stats import linregress
    from statsmodels.tsa.stattools import acf, adfuller

    try:
        # 1. Base details
        length = len(y)
        
        # safely extract frequency string
        freq = "Unknown"
        if hasattr(y.index, "freqstr") and y.index.freqstr:
            freq = y.index.freqstr
        elif hasattr(y.index, "freq") and y.index.freq is not None:
            if hasattr(y.index.freq, "freqstr"):
                freq = y.index.freq.freqstr
            else:
                freq = str(y.index.freq)

        # Ensure we have numeric data
        y_numeric = pd.to_numeric(y, errors="coerce").dropna()
        if len(y_numeric) < 10:
            return {
                "success": False,
                "error": "Time series too short for reliable statistical analysis (need >= 10 non-NaN values).",
            }

        # 2. Stationarity (ADF Test)
        try:
            adf_result = adfuller(y_numeric.values, autolag="AIC")
            p_value = float(adf_result[1])
            is_stationary = p_value < 0.05
            stationarity_stats = {
                "is_stationary": bool(is_stationary),
                "adf_statistic": float(adf_result[0]),
                "p_value": p_value,
                "critical_values": {k: float(v) for k, v in adf_result[4].items()},
            }
        except Exception as e:
            logger.warning(f"ADF test failed: {e}")
            stationarity_stats = {"error": str(e)}

        # 3. Presence of trend (Linear Regression)
        try:
            x = np.arange(len(y_numeric))
            slope, intercept, r_value, p_value_trend, std_err = linregress(x, y_numeric.values)
            has_trend = p_value_trend < 0.05 and abs(slope) > 1e-5
            trend_stats = {
                "has_trend": bool(has_trend),
                "slope": float(slope),
                "p_value": float(p_value_trend),
                "r_squared": float(r_value**2),
            }
        except Exception as e:
            logger.warning(f"Trend test failed: {e}")
            trend_stats = {"error": str(e)}

        # 4. Seasonality (ACF)
        try:
            nlags = min(40, len(y_numeric) // 2 - 1)
            if nlags > 0:
                acf_vals, confint = acf(y_numeric.values, nlags=nlags, alpha=0.05)

                # Identify significant peaks (outside confidence interval limits)
                # confint gives [lower, upper] bounds centered on the ACF value
                lower_bounds = confint[:, 0] - acf_vals
                upper_bounds = confint[:, 1] - acf_vals
                is_significant = (acf_vals > upper_bounds) | (acf_vals < lower_bounds)

                # Ignore lag 0 which is always 1 and significant
                is_significant[0] = False

                significant_lags = np.where(is_significant)[0]

                has_seasonality = len(significant_lags) > 0
                strongest_lag = None
                if has_seasonality:
                    # Find highest positive significant peak
                    positive_peaks = [lag for lag in significant_lags if acf_vals[lag] > 0]
                    if positive_peaks:
                        strongest_lag = int(positive_peaks[np.argmax([acf_vals[lag] for lag in positive_peaks])])
                    else:
                        strongest_lag = int(significant_lags[np.argmax(np.abs(acf_vals[significant_lags]))])

                seasonality_stats = {
                    "has_seasonality": bool(has_seasonality),
                    "strongest_seasonal_period": strongest_lag,
                    "significant_lags": [int(x) for x in significant_lags.tolist()],
                }
            else:
                seasonality_stats = {"error": "Not enough data for ACF"}
                
        except Exception as e:
            logger.warning(f"Seasonality test failed: {e}")
            seasonality_stats = {"error": str(e)}

        return {
            "success": True,
            "data_handle": data_handle,
            "summary": {
                "length": int(length),
                "frequency": freq,
                "missing_values": int(y.isna().sum()),
            },
            "stationarity": stationarity_stats,
            "trend": trend_stats,
            "seasonality": seasonality_stats,
        }

    except Exception as e:
        logger.exception("Error analyzing data")
        return {"success": False, "error": str(e)}
