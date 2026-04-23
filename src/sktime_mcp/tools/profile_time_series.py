import numpy as np
from typing import Any, Dict

def _autocorr(x, lag):
    x = np.asarray(x)
    if lag >= len(x):
        return 0.0
    x1 = x[:-lag]
    x2 = x[lag:]
    if x1.std() == 0 or x2.std() == 0:
        return 0.0
    return float(np.corrcoef(x1, x2)[0, 1])

def profile_time_series(y, freq: str | None = None) -> Dict[str, Any]:
    y = np.asarray(y, dtype=float)

    length = len(y)
    missing_ratio = float(np.isnan(y).mean())

    # basic stats (ignore NaN)
    y_clean = y[~np.isnan(y)]
    mean = float(np.mean(y_clean)) if len(y_clean) else 0.0
    std = float(np.std(y_clean)) if len(y_clean) else 0.0

    # simple trend proxy: corr(time, y)
    t = np.arange(length)
    if std == 0:
        trend_strength = 0.0
    else:
        trend_strength = float(np.corrcoef(t[~np.isnan(y)], y_clean)[0, 1])

    # simple seasonality proxy: autocorr at lag 12 (fallback 7)
    lag = 12 if length >= 24 else 7
    seasonal_strength = _autocorr(y_clean, lag) if len(y_clean) > lag else 0.0

    # crude stationarity proxy: mean change small + low trend
    diffs = np.diff(y_clean) if len(y_clean) > 1 else np.array([0.0])
    is_stationary = bool(abs(trend_strength) < 0.2 and np.std(diffs) < std)

    # simple horizon suggestion
    fh = list(range(1, min(13, max(2, length // 10))))

    return {
        "length": int(length),
        "freq": freq,
        "missing_ratio": missing_ratio,
        "mean": mean,
        "std": std,
        "trend_strength": trend_strength,
        "seasonal_strength": seasonal_strength,
        "is_stationary": is_stationary,
        "recommended_fh": fh,
    }