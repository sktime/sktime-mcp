"""
Diagnostic MCP tools for sktime-mcp.

Statistical preprocessing tools that give an LLM agent actionable signals
about the data before it decides on the next modeling step.

Implements:
- detect_seasonality: Detect cyclic patterns and quantify their strength.
- check_structural_break: Detect permanent regime changes using CUSUM.

See: https://github.com/sktime/sktime-mcp/issues/96
"""

from __future__ import annotations

import math
from typing import Any, Optional

import numpy as np


def _autocorrelation_fft(x: np.ndarray, max_lag: int) -> np.ndarray:
    """Compute autocorrelation using FFT for O(N log N) performance."""
    n = len(x)
    x_centered = x - x.mean()
    # Zero-pad to avoid circular correlation
    fft_size = 1
    while fft_size < 2 * n:
        fft_size *= 2
    fft_x = np.fft.rfft(x_centered, n=fft_size)
    acf_full = np.fft.irfft(fft_x * np.conj(fft_x), n=fft_size)[:n]
    # Normalize by variance
    if acf_full[0] == 0:
        return np.zeros(min(max_lag, n - 1))
    acf_normalized = acf_full / acf_full[0]
    return acf_normalized[1 : max_lag + 1]


# Known seasonal periods for common frequencies
_FREQ_PERIODS: dict[str, list[int]] = {
    "D": [7, 14, 28, 30, 365],
    "W": [4, 13, 26, 52],
    "M": [3, 6, 12],
    "Q": [4],
    "H": [24, 168],
    "T": [60, 1440],
    "min": [60, 1440],
}


def detect_seasonality_tool(
    data: list[float],
    frequency: Optional[str] = None,
    max_lag: Optional[int] = None,
) -> dict[str, Any]:
    """Detect cyclic patterns in a time series and quantify their strength.

    Uses FFT-accelerated autocorrelation with adaptive detrending and a
    dynamic significance threshold.

    Args:
        data: The time series values as a flat list of floats.
        frequency: Optional frequency hint (e.g. "D", "W", "M") to boost
            known seasonal periods during candidate scoring.
        max_lag: Maximum lag to check. Defaults to min(len(data)//2, 200).

    Returns:
        Dictionary with seasonality analysis results including period,
        strength, confidence, candidate periods, and a next_action_hint.
    """
    try:
        x = np.asarray(data, dtype=np.float64)
        if not np.all(np.isfinite(x)):
            return {
                "success": False,
                "error": "Series contains NaN or Inf values. Remove or interpolate them before analysis.",
            }
        if len(x) < 6:
            return {
                "success": False,
                "error": "Series too short for seasonality detection (need >= 6 points)",
            }

        n = len(x)
        if max_lag is None:
            max_lag = min(n // 2, 200)

        # Adaptive detrending: apply first-order differencing only if it
        # reduces variance (prevents trend from faking seasonality in ACF)
        x_diff = np.diff(x)
        if np.var(x_diff) < np.var(x):
            x_for_acf = x_diff
            method = "autocorrelation_detrended_fft"
        else:
            x_for_acf = x
            method = "autocorrelation_fft"

        # Compute ACF via FFT
        effective_max_lag = min(max_lag, len(x_for_acf) - 1)
        if effective_max_lag < 2:
            return {
                "success": True,
                "period": None,
                "strength": 0.0,
                "seasonality_class": "none",
                "confidence": "low",
                "candidates": [],
                "method": method,
                "next_action_hint": "no_seasonal_pattern",
            }

        acf = _autocorrelation_fft(x_for_acf, effective_max_lag)

        # Dynamic significance threshold
        significance = max(1.96 / math.sqrt(n), 0.2)

        # Find candidate peaks above significance threshold
        candidates = []
        for lag in range(1, len(acf)):
            corr = float(acf[lag])
            if corr > significance:
                # Check if it's a local peak
                is_peak = True
                if lag > 1 and acf[lag - 1] >= corr:
                    is_peak = False
                if lag < len(acf) - 1 and acf[lag + 1] >= corr:
                    is_peak = False
                if is_peak:
                    score = corr
                    # Frequency-aware scoring bonus
                    if frequency and frequency in _FREQ_PERIODS:
                        if (lag + 1) in _FREQ_PERIODS[frequency]:
                            score *= 1.1
                    candidates.append({
                        "period": lag + 1,
                        "correlation": round(corr, 4),
                        "score": round(score, 4),
                    })

        # Sort by score descending
        candidates.sort(key=lambda c: c["score"], reverse=True)
        # Keep top 5
        candidates = candidates[:5]
        # Remove internal score from output
        for c in candidates:
            del c["score"]

        if not candidates:
            return {
                "success": True,
                "period": None,
                "strength": 0.0,
                "seasonality_class": "none",
                "confidence": "low",
                "candidates": [],
                "method": method,
                "next_action_hint": "no_seasonal_pattern",
            }

        best = candidates[0]
        strength = best["correlation"]

        # Classify strength
        if strength >= 0.7:
            seasonality_class = "strong"
            confidence = "high"
        elif strength >= 0.4:
            seasonality_class = "moderate"
            confidence = "medium"
        else:
            seasonality_class = "weak"
            confidence = "low"

        # Determine next action hint
        if seasonality_class in ("strong", "moderate"):
            next_action = "deseasonalize_then_stationarity"
        else:
            next_action = "check_stationarity_directly"

        return {
            "success": True,
            "period": best["period"],
            "strength": round(strength, 4),
            "seasonality_class": seasonality_class,
            "confidence": confidence,
            "candidates": candidates,
            "method": method,
            "next_action_hint": next_action,
        }

    except Exception as e:
        return {"success": False, "error": str(e)}


def check_structural_break_tool(
    data: list[float],
) -> dict[str, Any]:
    """Detect permanent regime changes in a time series using CUSUM.

    Uses retrospective global CUSUM on raw (non-differenced) data with a
    dynamic threshold based on a Brownian Bridge confidence bound.

    Args:
        data: The time series values as a flat list of floats.

    Returns:
        Dictionary with break detection results including location,
        confidence, test statistic, and a next_action_hint.
    """
    try:
        x = np.asarray(data, dtype=np.float64)
        if not np.all(np.isfinite(x)):
            return {
                "success": False,
                "error": "Series contains NaN or Inf values. Remove or interpolate them before analysis.",
            }
        if len(x) < 10:
            return {
                "success": False,
                "error": "Series too short for structural break detection (need >= 10 points)",
            }

        n = len(x)
        mean = x.mean()
        std = x.std(ddof=1)

        if std == 0:
            return {
                "success": True,
                "break_detected": False,
                "location": None,
                "confidence": 0.0,
                "test_stat": 0.0,
                "next_action_hint": "constant_series",
            }

        # Retrospective global CUSUM: cumulative sum of mean-centered data
        cusum = np.cumsum(x - mean)

        # Break point = index of maximum absolute divergence
        abs_cusum = np.abs(cusum)
        break_idx = int(np.argmax(abs_cusum))

        # Test statistic: max divergence normalized by std * sqrt(N)
        test_stat = float(abs_cusum[break_idx] / (std * math.sqrt(n)))

        # Dynamic threshold: 1.358 * (1 + 0.1 * ln(N))
        # 1.358 base = 95% confidence bound of Brownian Bridge
        # Log-scaled adjustment prevents long series from false positives
        threshold = 1.358 * (1.0 + 0.1 * math.log(n))

        break_detected = test_stat > threshold

        # Confidence: how far above the threshold (capped at 1.0)
        if threshold > 0:
            confidence = min(test_stat / threshold, 2.0) / 2.0
        else:
            confidence = 0.0

        if break_detected:
            next_action = "get_dataset_history"
        else:
            next_action = "proceed_with_full_series"

        location_fraction = round(break_idx / n, 4) if break_detected else None

        return {
            "success": True,
            "break_detected": break_detected,
            "location": break_idx if break_detected else None,
            "location_fraction": location_fraction,
            "confidence": round(confidence, 4),
            "test_stat": round(test_stat, 4),
            "next_action_hint": next_action,
        }

    except Exception as e:
        return {"success": False, "error": str(e)}
