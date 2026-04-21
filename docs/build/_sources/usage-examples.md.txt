# Usage Examples

This page provides common usage patterns and example flows for interacting with the sktime MCP server. These examples demonstrate how an LLM or user can interact with the server to perform time series forecasting tasks.

## 🔄 Example LLM Flows

These flows show how an LLM would chain multiple tool calls to achieve a high-level goal.

### Flow 1: Simple Forecasting

**User Prompt:** "Forecast monthly airline passengers using a probabilistic model."

The agent begins by searching for forecasting models that support the required capabilities.

**1. Discover Models**
First, the agent searches for forecasting models that support prediction intervals.

```json
{
  "name": "list_estimators",
  "arguments": {
    "task": "forecasting",
    "tags": {
      "capability:pred_int": true
    }
  }
}
```

After identifying suitable models, the agent examines a specific estimator to understand how it can be configured.

**2. Inspect Choice**
The agent inspects a specific estimator (e.g., ARIMA) to understand its parameters.

```json
{
  "name": "describe_estimator",
  "arguments": {
    "estimator": "ARIMA"
  }
}
```

With the estimator selected and understood, the agent creates a concrete instance that can be fitted to data.

**3. Instantiate**
The agent instantiates the estimator with specific parameters.

```json
{
  "name": "instantiate_estimator",
  "arguments": {
    "estimator": "ARIMA",
    "params": {
      "order": [1, 1, 1]
    }
  }
}
```
*Returns:* `{"handle": "est_abc123"}`

**4. Execute**
Finally, the agent fits the model on the airline dataset and generates a forecast.

```json
{
  "name": "fit_predict",
  "arguments": {
    "estimator_handle": "est_abc123",
    "dataset": "airline",
    "horizon": 12
  }
}
```
*Returns:* `{"predictions": {1: 450.2, 2: 455.1, ...}}`

---

### Flow 2: Pipeline Forecasting ⭐

**User Prompt:** "Forecast with deseasonalization and detrending preprocessing."

**1. Validate Composition**
Before creating the pipeline, the agent checks if the components can work together.

```json
{
  "name": "validate_pipeline",
  "arguments": {
    "components": ["ConditionalDeseasonalizer", "Detrender", "ARIMA"]
  }
}
```
*Returns:* `{"valid": true}`

**2. Instantiate Pipeline**
The agent instantiates the entire pipeline in a single call.

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

**3. Execute**
The pipeline is executed just like a single estimator.

```json
{
  "name": "fit_predict",
  "arguments": {
    "estimator_handle": "est_xyz789",
    "dataset": "airline",
    "horizon": 12
  }
}
```

## 🛠️ Individual Tool Usage

Here are some standalone examples of using specific tools.

### Discovery & Search

**List Estimators**
```json
{
  "name": "list_estimators",
  "arguments": {
    "task": "forecasting",
    "tags": {
      "capability:pred_int": true
    },
    "limit": 10
  }
}
```

**Search by Name**
```json
{
  "name": "search_estimators",
  "arguments": {
    "query": "ARIMA",
    "limit": 5
  }
}
```

### Handle Management

**Release a Handle**
When you are done with an estimator, it's good practice to release it to free up resources.

```json
{
  "name": "release_handle",
  "arguments": {
    "handle": "est_abc123"
  }
}
```

### Model Persistence

**Save a Fitted Estimator**
Use `save_model` after fitting an estimator or pipeline handle. The underlying `sktime.utils.mlflow_sktime.save_model` API saves to a local filesystem path.

```json
{
  "name": "save_model",
  "arguments": {
    "estimator_handle": "est_abc123",
    "path": "/absolute/path/to/model_dir",
    "mlflow_params": {
      "serialization_format": "pickle"
    }
  }
}
```

*Returns:* `{"success": true, "saved_path": "/absolute/path/to/model_dir", "message": "Model saved successfully to '/absolute/path/to/model_dir'"}`
