"""Tool for recommending appropriate forecasters based on data characteristics."""

from typing import Any, Optional


def recommend_forecaster(
    data_characteristics: Optional[dict[str, Any]] = None,
    requirements: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """
    Recommend appropriate forecasting models based on data characteristics and requirements.

    Parameters
    ----------
    data_characteristics : dict, optional
        Dictionary containing data characteristics such as:
        - 'frequency': str, data frequency (e.g., 'daily', 'monthly', 'hourly')
        - 'length': int, number of observations
        - 'seasonality': bool, whether data shows seasonal patterns
        - 'trend': bool, whether data shows trend
        - 'multivariate': bool, whether data is multivariate
        - 'missing_values': bool, whether data contains missing values
    requirements : dict, optional
        Dictionary containing user requirements such as:
        - 'interpretability': str, importance level ('high', 'medium', 'low')
        - 'speed': str, importance level ('high', 'medium', 'low')
        - 'accuracy': str, importance level ('high', 'medium', 'low')
        - 'forecast_horizon': int, number of steps to forecast

    Returns
    -------
    dict
        Dictionary containing:
        - 'recommendations': list of recommended forecaster names with rationale
        - 'alternatives': list of alternative forecasters
        - 'warnings': list of potential issues or considerations
    """
    data_characteristics = data_characteristics or {}
    requirements = requirements or {}

    recommendations = []
    alternatives = []
    warnings = []

    # Extract characteristics
    length = data_characteristics.get("length", 0)
    seasonality = data_characteristics.get("seasonality", False)
    trend = data_characteristics.get("trend", False)
    multivariate = data_characteristics.get("multivariate", False)
    missing_values = data_characteristics.get("missing_values", False)

    # Extract requirements
    interpretability = requirements.get("interpretability", "medium")
    speed = requirements.get("speed", "medium")
    accuracy = requirements.get("accuracy", "high")

    # Recommendation logic
    if length < 30:
        warnings.append("Limited data available. Consider simple models or collect more data.")
        recommendations.append({
            "name": "NaiveForecaster",
            "rationale": "Simple baseline suitable for limited data",
        })
        alternatives.append("ExponentialSmoothing")
    elif seasonality and trend:
        if accuracy == "high":
            recommendations.append({
                "name": "AutoARIMA",
                "rationale": "Handles both seasonality and trend automatically",
            })
            alternatives.extend(["Prophet", "TBATS"])
        else:
            recommendations.append({
                "name": "ExponentialSmoothing",
                "rationale": "Fast and effective for seasonal data with trend",
            })
    elif seasonality:
        recommendations.append({
            "name": "SeasonalNaive",
            "rationale": "Simple and effective for seasonal patterns",
        })
        alternatives.append("SARIMAX")
    elif trend:
        recommendations.append({
            "name": "TrendForecaster",
            "rationale": "Specialized for trending data",
        })
        alternatives.append("AutoARIMA")
    else:
        recommendations.append({
            "name": "NaiveForecaster",
            "rationale": "Simple baseline for data without clear patterns",
        })

    if multivariate:
        recommendations.append({
            "name": "VAR",
            "rationale": "Designed for multivariate time series",
        })
        alternatives.append("VectorARIMA")

    if missing_values:
        warnings.append(
            "Data contains missing values. Consider imputation or models that handle missing data."
        )

    if interpretability == "high":
        warnings.append(
            "For high interpretability, prefer ARIMA, ExponentialSmoothing, or Naive methods."
        )

    if speed == "high" and accuracy == "high":
        warnings.append("High speed and high accuracy may be conflicting requirements.")

    return {
        "recommendations": recommendations,
        "alternatives": alternatives,
        "warnings": warnings,
    }


def recommend_forecaster_tool(
    data_characteristics: Optional[str] = None,
    requirements: Optional[str] = None,
) -> dict[str, Any]:
    """
    MCP tool wrapper for recommending forecasters.

    Parameters
    ----------
    data_characteristics : str, optional
        JSON string containing data characteristics
    requirements : str, optional
        JSON string containing user requirements

    Returns
    -------
    dict
        Dictionary containing recommendations, alternatives, and warnings
    """
    import json

    data_chars = json.loads(data_characteristics) if data_characteristics else {}
    reqs = json.loads(requirements) if requirements else {}

    return recommend_forecaster(data_characteristics=data_chars, requirements=reqs)