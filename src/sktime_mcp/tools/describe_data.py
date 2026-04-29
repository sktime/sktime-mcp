"""describe_data_tool — series fingerprinting for agentic model selection.

Returns summary statistics so an LLM agent can reason about data
characteristics (trend, seasonality, missingness) before choosing a
forecaster. Addresses the gap described in:
https://github.com/sktime/sktime-mcp/issues/<YOUR_ISSUE_NUMBER>
"""

from __future__ import annotations

from typing import Any

import numpy as np


def describe_data_tool(
    dataset: str,
    target_col: str | None = None,
) -> dict[str, Any]:
    """Return a statistical fingerprint of a named sktime dataset.

    Parameters
    ----------
    dataset : str
        Dataset name — same values accepted by evaluate_estimator
        (e.g. "airline", "sunspots", "lynx", "shampoo_sales").
    target_col : str, optional
        Column to describe for multivariate datasets. Ignored for
        univariate series.

    Returns
    -------
    dict with keys:
        success, length, n_missing, min, max, mean, std,
        trend_slope_per_step, candidate_seasonal_period, frequency
    """
    try:
        from sktime.datasets import load_airline, load_longley, load_lynx, load_sunspots  # type: ignore

        _LOADERS: dict[str, Any] = {
            "airline": load_airline,
            "sunspots": load_sunspots,
            "lynx": load_lynx,
        }

        import pandas as pd

        if dataset in _LOADERS:
            y = _LOADERS[dataset]()
        else:
            # Best-effort: try sktime's generic loader
            try:
                from sktime.datasets import load_from_tsfile_to_dataframe  # type: ignore
                y = load_from_tsfile_to_dataframe(dataset)
            except Exception:
                return {
                    "success": False,
                    "error": (
                        f"Unknown dataset '{dataset}'. "
                        "Known values: airline, sunspots, lynx. "
                        "Use list_available_data to see all options."
                    ),
                }

        # Flatten to 1-D numpy array
        if isinstance(y, pd.DataFrame):
            col = target_col or y.columns[0]
            arr = np.asarray(y[col], dtype=float)
        else:
            arr = np.asarray(y, dtype=float)

        finite = arr[np.isfinite(arr)]

        result: dict[str, Any] = {
            "success": True,
            "dataset": dataset,
            "length": int(len(arr)),
            "n_missing": int(np.sum(~np.isfinite(arr))),
            "min": round(float(np.nanmin(arr)), 4) if finite.size else None,
            "max": round(float(np.nanmax(arr)), 4) if finite.size else None,
            "mean": round(float(np.nanmean(arr)), 4) if finite.size else None,
            "std": round(float(np.nanstd(arr)), 4) if finite.size else None,
            "trend_slope_per_step": _trend_slope(arr),
            "candidate_seasonal_period": _detect_seasonality(arr),
            "frequency": None,
        }

        if isinstance(y, pd.Series) and isinstance(y.index, pd.DatetimeIndex):
            try:
                result["frequency"] = pd.infer_freq(y.index)
            except Exception:
                pass

        return result

    except Exception as exc:
        return {"success": False, "error": str(exc)}


# --------------------------------------------------------------------------- #
# Internal helpers
# --------------------------------------------------------------------------- #


def _trend_slope(arr: np.ndarray) -> float | None:
    """OLS slope of finite values vs index — quick trend estimate."""
    mask = np.isfinite(arr)
    if mask.sum() < 3:
        return None
    idx = np.arange(len(arr), dtype=float)
    try:
        slope, _ = np.polyfit(idx[mask], arr[mask], 1)
        return round(float(slope), 6)
    except Exception:
        return None


def _detect_seasonality(arr: np.ndarray) -> int | None:
    """Detect seasonal period via ACF of the first-differenced series.

    First-differencing removes linear trend before computing ACF, so
    seasonal peaks are not masked by trend autocorrelation (a known
    failure mode of raw-ACF approaches on trending data such as airline
    passengers).
    """
    if len(arr) < 24:
        return None
    arr = np.nan_to_num(arr, nan=float(np.nanmean(arr)))
    d = np.diff(arr)
    d = d - d.mean()
    n = len(d)
    max_lag = min(60, n // 2)
    if max_lag < 4:
        return None
    denom = float(np.dot(d, d))
    if denom == 0:
        return None
    acfs = [
        (lag, float(np.dot(d[:-lag], d[lag:]) / denom))
        for lag in range(2, max_lag + 1)
    ]
    best_lag, best_acf = max(acfs, key=lambda t: t[1])
    return best_lag if best_acf > 0.2 else None