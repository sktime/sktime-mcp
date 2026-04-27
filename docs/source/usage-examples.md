# Usage Examples

This page shows concrete examples of what you can ask your AI assistant to do with sktime-mcp, what to expect in return, and — for those interested — the MCP tool calls that happen behind the scenes.

---

## 🔄 End-to-End Examples

### Example 1: Simple Forecasting

**What you type:**

> *"Forecast monthly airline passengers for the next 12 months using a probabilistic model."*

**What happens:**

1. The assistant searches sktime's registry for forecasting models that support prediction intervals.
2. It picks a suitable model (e.g., ARIMA) and shows you the options, or asks you to choose.
3. Once confirmed, it instantiates the model, fits it on the `airline` demo dataset, and generates a 12-month forecast.
4. You receive the predicted values (and optionally prediction intervals) in the conversation.

**What you get back (example):**

> Here are the forecasted airline passenger counts for the next 12 months:
>
> | Month | Forecast | Lower 95% | Upper 95% |
> |-------|----------|-----------|-----------|
> | 1     | 450.2    | 420.1     | 480.3     |
> | 2     | 455.1    | 418.7     | 491.5     |
> | ...   | ...      | ...       | ...       |
>
> The model used was ARIMA(1,1,1) fitted on the `airline` dataset (144 observations).

**Follow-up prompts you could try:**

- *"Try the same forecast with ExponentialSmoothing instead."*
- *"Show me the Python code to reproduce this."*
- *"Plot the forecast against the historical data."*

<details>
<summary>🔧 MCP tool calls (what happens under the hood)</summary>

**1. Discover models with prediction interval support:**
```json
{
  "name": "list_estimators",
  "arguments": {
    "task": "forecasting",
    "tags": {"capability:pred_int": true}
  }
}
```

**2. Inspect the chosen estimator:**
```json
{
  "name": "describe_estimator",
  "arguments": {"estimator": "ARIMA"}
}
```

**3. Instantiate with specific parameters:**
```json
{
  "name": "instantiate_estimator",
  "arguments": {"estimator": "ARIMA", "params": {"order": [1, 1, 1]}}
}
```
*Returns:* `{"handle": "est_abc123"}`

**4. Fit and predict:**
```json
{
  "name": "fit_predict",
  "arguments": {"estimator_handle": "est_abc123", "dataset": "airline", "horizon": 12}
}
```

</details>

---

### Example 2: Pipeline Forecasting

**What you type:**

> *"I want to forecast with deseasonalization and detrending as preprocessing. Use ARIMA as the final model."*

**What happens:**

1. The assistant validates that `ConditionalDeseasonalizer → Detrender → ARIMA` is a valid pipeline in sktime.
2. It creates the pipeline and fits it on the data.
3. Predictions reflect the full preprocessing chain — deseasonalized, detrended, then forecast, then inverse-transformed back to the original scale.

**What you get back:**

> Pipeline created: **ConditionalDeseasonalizer → Detrender → ARIMA(1,1,1)**
>
> Forecast for the next 12 months on the `airline` dataset:
>
> | Month | Forecast |
> |-------|----------|
> | 1     | 448.7    |
> | 2     | 461.3    |
> | ...   | ...      |
>
> The preprocessing steps (deseasonalization and detrending) were applied automatically.

**Follow-up prompts you could try:**

- *"Replace ARIMA with ExponentialSmoothing in this pipeline."*
- *"Evaluate this pipeline with cross-validation."*
- *"What other preprocessing transformers are available?"*

<details>
<summary>🔧 MCP tool calls (what happens under the hood)</summary>

**1. Validate composition:**
```json
{
  "name": "validate_pipeline",
  "arguments": {"components": ["ConditionalDeseasonalizer", "Detrender", "ARIMA"]}
}
```
*Returns:* `{"valid": true}`

**2. Instantiate pipeline:**
```json
{
  "name": "instantiate_pipeline",
  "arguments": {
    "components": ["ConditionalDeseasonalizer", "Detrender", "ARIMA"],
    "params_list": [{}, {}, {"order": [1, 1, 1]}]
  }
}
```
*Returns:* `{"handle": "est_xyz789", "pipeline": "ConditionalDeseasonalizer → Detrender → ARIMA"}`

**3. Execute:**
```json
{
  "name": "fit_predict",
  "arguments": {"estimator_handle": "est_xyz789", "dataset": "airline", "horizon": 12}
}
```

</details>

---

### Example 3: Forecasting on Your Own CSV

**What you type:**

> *"Load my sales data from /home/user/sales.csv — the date column is 'month' and the target is 'revenue'. Then forecast 6 months ahead with AutoARIMA."*

**What happens:**

1. The assistant loads your CSV, using the columns you specified.
2. It auto-formats the data (infers frequency, handles missing values if needed) and reports a summary.
3. It instantiates `AutoARIMA`, fits it on your data, and returns a 6-month forecast.

**What you get back:**

> Loaded **sales.csv**: 48 rows, monthly frequency (2020-01 to 2023-12), target column "revenue".
>
> AutoARIMA selected order (2,1,1) with seasonal order (1,1,0,12).
>
> Forecast:
>
> | Month   | Revenue Forecast |
> |---------|-----------------|
> | 2024-01 | 12,450          |
> | 2024-02 | 13,100          |
> | ...     | ...             |

<details>
<summary>🔧 MCP tool calls (what happens under the hood)</summary>

**1. Load data:**
```json
{
  "name": "load_data_source",
  "arguments": {
    "config": {
      "type": "file",
      "path": "/home/user/sales.csv",
      "time_column": "month",
      "target_column": "revenue"
    }
  }
}
```
*Returns:* `{"handle": "data_abc"}`

**2. Instantiate AutoARIMA:**
```json
{
  "name": "instantiate_estimator",
  "arguments": {"estimator": "AutoARIMA"}
}
```
*Returns:* `{"handle": "est_def456"}`

**3. Fit and predict:**
```json
{
  "name": "fit_predict",
  "arguments": {"estimator_handle": "est_def456", "data_handle": "data_abc", "horizon": 6}
}
```

</details>

---

### Example 4: Model Comparison

**What you type:**

> *"Compare NaiveForecaster, ExponentialSmoothing, and ARIMA on the airline dataset using cross-validation."*

**What happens:**

1. The assistant instantiates each model.
2. For each model, it runs cross-validated evaluation (expanding window backtest).
3. It presents a comparison table of error metrics so you can pick the best model.

**What you get back:**

> Cross-validation results on `airline` (3 folds, expanding window):
>
> | Model                  | Mean Absolute Error | RMSE   |
> |------------------------|--------------------:|-------:|
> | NaiveForecaster        | 38.2                | 45.1   |
> | ExponentialSmoothing   | 24.7                | 31.3   |
> | ARIMA(1,1,1)           | 22.1                | 28.9   |
>
> **ARIMA(1,1,1)** performed best on this dataset.

**Follow-up prompts:**

- *"Use ARIMA then — forecast 24 months ahead."*
- *"Save the ARIMA model for later."*
- *"Export the comparison code as a Python script."*

---

## 🛠️ Individual Tasks

Below are standalone tasks you can ask about at any time — they don't need to be part of a larger workflow.

### Discovering Models

> *"What classification models does sktime have?"*
>
> *"Show me forecasters that handle missing data."*
>
> *"Tell me about the ThetaForecaster — what parameters does it take?"*

### Loading and Inspecting Data

> *"What demo datasets are available?"*
>
> *"Load my data from /home/user/experiment.parquet with time column 'ts' and target 'measurement'."*
>
> *"Describe the data I just loaded — how many rows, what's the frequency?"*

### Exporting and Saving

> *"Give me the Python code for the pipeline I just built."*
>
> *"Save this fitted model to /home/user/models/best_forecaster."*

### Cleanup

> *"Release all active handles to free up memory."*
>
> *"What handles are currently active?"*

---

## 📋 MCP Tool Reference

For developers and advanced users who want to understand the full MCP tool signatures, see the [Ideal MCP Tools](ideal-mcp-tools.md) design document. That page documents every tool's input schema, parameters, and intended use cases.
