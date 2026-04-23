import pandas as pd
from statsmodels.tsa.stattools import adfuller
from sktime.utils.seasonality import autocorrelation_seasonality_test


def get_timeseries_diagnostics(series: list):

    series = pd.Series(series)

    diagnostics = {}

    diagnostics["length"] = len(series)
    diagnostics["missing_values"] = int(series.isna().sum())

    # stationarity
    try:
        adf = adfuller(series.dropna())
        diagnostics["is_stationary"] = adf[1] < 0.05
        diagnostics["adf_p_value"] = float(adf[1])
    except Exception:
        diagnostics["is_stationary"] = None
        diagnostics["adf_p_value"] = None

    # seasonality
    try:
        seasonal = autocorrelation_seasonality_test(series.dropna(), sp=12)
        diagnostics["is_seasonal"] = bool(seasonal)
        diagnostics["seasonal_period"] = 12 if seasonal else None
    except Exception:
        diagnostics["is_seasonal"] = None
        diagnostics["seasonal_period"] = None

    stats = series.describe()

    diagnostics["stats"] = {
        "mean": float(stats["mean"]),
        "std": float(stats["std"]),
        "min": float(stats["min"]),
        "max": float(stats["max"]),
    }

    diagnostics["model_hints"] = {}

    if diagnostics["is_seasonal"]:
        diagnostics["model_hints"]["recommended_models"] = ["SARIMA", "ETS"]
    else:
        diagnostics["model_hints"]["recommended_models"] = ["ARIMA"]

    if diagnostics["is_stationary"] is False:
        diagnostics["model_hints"]["needs_differencing"] = True

    return diagnostics