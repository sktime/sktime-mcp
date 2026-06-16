# Agent Workflow Test Notes

This page records small end-to-end checks of `sktime-mcp` from an agent workflow
perspective. The goal is to show which MCP tool chains work today and where an
assistant may need clearer tool guidance.

## Basic Forecasting Workflow

**Prompt shape tested:**

> Forecast the airline dataset for six periods with a simple baseline forecaster,
> then export reproducible Python code.

**Tool chain tested:**

1. `list_available_data(is_demo=true)`
2. `list_estimators(task="forecasting", query="NaiveForecaster", limit=5)`
3. `instantiate_estimator(estimator="NaiveForecaster", params={"strategy": "last"})`
4. `fit_predict(estimator_handle=<handle>, dataset="airline", horizon=6)`
5. `export_code(handle=<handle>, var_name="forecaster", include_fit_example=true, dataset="airline")`

**Observed result:**

- Demo data discovery succeeded and included the `airline` dataset.
- Estimator instantiation succeeded for `NaiveForecaster(strategy="last")`.
- `fit_predict` succeeded for a six-step horizon.
- The forecast returned six monthly values, with the first three values:

```text
1961-01: 432.0
1961-02: 432.0
1961-03: 432.0
```

- Code export succeeded and produced a reproducible snippet beginning with:

```python
from sktime.forecasting.naive._naive import NaiveForecaster

forecaster = NaiveForecaster(strategy="last")

# Example usage:
# Load data
from sktime.datasets import load_airline
y = load_airline()
```

## Agent-Facing Notes

The core forecasting path works for a simple baseline model: an assistant can
discover data, instantiate an estimator, run a forecast, and export code without
manual Python scripting.

One discovery caveat appeared during the run. Searching
`list_estimators(task="forecasting", query="NaiveForecaster", limit=5)` returned
valid forecasting results, but the first page contained broader matches such as
ensemble and interval forecasters rather than the exact `NaiveForecaster` class.
The workflow still succeeded once the exact estimator name was supplied to
`instantiate_estimator`, but agents may need guidance to increase `limit`, inspect
additional pages, or instantiate directly when the exact class name is already
known.

This suggests two useful follow-ups for agent reliability:

- Improve exact-name ranking in estimator search results.
- Add examples that show when agents should use direct instantiation after a
  known estimator name has been identified.
