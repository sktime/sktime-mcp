# Usage Examples

This page provides common usage patterns and example flows for interacting with the sktime MCP server. These examples demonstrate how an LLM or user can interact with the server to perform time series forecasting tasks.

## 🔄 Example LLM Flows

These flows show how an LLM would chain multiple tool calls to achieve a high-level goal.

Each tool call returns information or a handle that the next step depends on, so the workflow is built step by step.

### Flow 1: Simple Forecasting

**User Prompt:** "Forecast monthly airline passengers using a probabilistic model."

The agent begins by searching for forecasting models that support the required capabilities.

**1. Discover Models**
First, the agent searches for forecasting models that support prediction intervals.

```json
{
  "name": "list_estimators",   // Step 1: discover forecasting models
  "arguments": {
    "task": "forecasting",
    "tags": {
      "capability:pred_int": true   // filter for probabilistic forecasters
    }
  }
}
```

After identifying suitable models, the agent examines a specific estimator to understand how it can be configured.

**2. Inspect Choice**
The agent inspects a specific estimator (e.g., ARIMA) to understand its parameters.

```json
{
  "name": "describe_estimator",   // Inspect the chosen estimator in detail
  "arguments": {
    "estimator": "ARIMA"   // The estimator we want more information about
  }
}
```

With the estimator selected and understood, the agent creates a concrete instance that can be fitted to data.

**3. Instantiate**
The agent instantiates the estimator with specific parameters.

```json
{
  "name": "instantiate_estimator",   // Create a concrete model instance
  "arguments": {
    "estimator": "ARIMA",   // The estimator we want to instantiate
    "params": {
      "order": [1, 1, 1]   // ARIMA(p, d, q) configuration
    }
  }
}
```
*Returns:* `{"handle": "est_abc123"}`

Handles like `est_abc123` represent live estimator objects stored in the MCP runtime and reused across tool calls.

**4. Execute**
Finally, the agent fits the model on the airline dataset and generates a forecast.

```json
{
  "name": "fit_predict",   // Fit the model and generate a forecast
  "arguments": {
    "estimator_handle": "est_abc123",   // handle returned from instantiate_estimator
    "dataset": "airline",   // built‑in demo dataset
    "horizon": 12   // forecast 12 future periods
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
  "name": "validate_pipeline",   // Check if these components can be chained together
  "arguments": {
    "components": ["ConditionalDeseasonalizer", "Detrender", "ARIMA"]   // Order of components in the pipeline
  }
}
```
*Returns:* `{"valid": true}`

**2. Instantiate Pipeline**
The agent instantiates the entire pipeline in a single call.

```json
{
  "name": "instantiate_pipeline",   // Build the full preprocessing + model pipeline
  "arguments": {
    "components": ["ConditionalDeseasonalizer", "Detrender", "ARIMA"],   // Pipeline steps in order
    "params_list": [{}, {}, {"order": [1, 1, 1]}]   // Parameters for each component (aligned by index)
  }
}
```
*Returns:* `{"handle": "est_xyz789", "pipeline": "ConditionalDeseasonalizer → Detrender → ARIMA"}`

**3. Execute**
The pipeline is executed just like a single estimator.

```json
{
  "name": "fit_predict",   // Fit the full pipeline and generate a forecast
  "arguments": {
    "estimator_handle": "est_xyz789",   // handle returned from instantiate_pipeline
    "dataset": "airline",   // built‑in demo dataset
    "horizon": 12   // forecast 12 future periods
  }
}
```

## 🛠️ Individual Tool Usage

Here are some standalone examples of using specific tools.

### Discovery & Search

**List Estimators**
```json
{
  "name": "list_estimators",   // List estimators matching a task and optional tags
  "arguments": {
    "task": "forecasting",   // Type of task we are interested in
    "tags": {
      "capability:pred_int": true   // Only return models that support prediction intervals
    },
    "limit": 10   // Maximum number of results to return
  }
}
```

**Search by Name**
```json
{
  "name": "search_estimators",   // Search estimators by name or keyword
  "arguments": {
    "query": "ARIMA",   // Search term to match estimator names/descriptions
    "limit": 5   // Maximum number of results to return
  }
}
```

### Handle Management

**Release a Handle**
When you are done with an estimator, it's good practice to release it to free up resources.

```json
{
  "name": "release_handle",   // Free the estimator handle from the MCP runtime
  "arguments": {
    "handle": "est_abc123"   // The handle we previously created and no longer need
  }
}
```

### Model Persistence

**Save a Fitted Estimator**
Use `save_model` after fitting an estimator or pipeline handle. The underlying `sktime.utils.mlflow_sktime.save_model` API saves to a local filesystem path.

```json
{
  "name": "save_model",   // Persist a fitted estimator or pipeline to disk
  "arguments": {
    "estimator_handle": "est_abc123",   // Handle of the fitted estimator or pipeline
    "path": "/absolute/path/to/model_dir",   // Local directory where the model will be saved
    "mlflow_params": {
      "serialization_format": "pickle"   // Serialization format used by MLflow
    }
  }
}
```

*Returns:* `{"success": true, "saved_path": "/absolute/path/to/model_dir", "message": "Model saved successfully to '/absolute/path/to/model_dir'"}`   // Confirmation that the model was persisted
