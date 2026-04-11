# 📘 User Guide

Welcome to the **sktime-mcp** User Guide. This comprehensive manual will help you understand how to install, configure, and master the Model Context Protocol (MCP) server for time-series forecasting.

---

## 🚀 Getting Started

### Prerequisites

Before you begin, ensure you have:

- **Python 3.9+** installed.
- **pip** package manager.
- A compatible MCP Client (like **Claude Desktop**).

### Installation

Install the package directly from the source. We recommend installing with all dependencies to unlock full functionality.

```bash
# Standard installation
pip install -e .

# Recommended: Install with all optional extras (SQL, Forecasting, Files)
pip install -e ".[all]"
```

### Running the Server

Start the MCP server to begin listening for connections:

```bash
sktime-mcp
```

*Or manually via Python:*
```bash
python -m sktime_mcp.server
```

!!! tip "Client Configuration"
    Ensure your MCP client (e.g., Claude Desktop) is configured to run this command. See the [official VSCode guidelines](https://code.visualstudio.com/docs/copilot/customization/mcp-servers#_configure-the-mcpjson-file) for configuration examples.

---

## 🛠️ Core Capabilities

The `sktime-mcp` server exposes a suite of tools designed for Large Language Models to interact with time-series data.

| Category | Tools | Description |
|----------|-------|-------------|
| **Discovery** | `list_estimators`, `search_estimators`, `describe_estimator` | Find the right model for your task (Forecasting, Classification, etc.). |
| **Instantiation** | `instantiate_estimator`, `instantiate_pipeline` | Create model instances or complex pipelines. |
| **Execution** | `fit_predict`, `fit_predict_async`, `fit_predict_with_data` | Train models and generate forecasts on demo or user data. |
| **Data** | `load_data_source`, `list_available_data` | Load data from Pandas, CSV/Parquet, or SQL, and inspect demo datasets plus active handles. |
| **Export** | `export_code`, `save_model` | Generate Python code or persist fitted estimators to a local path. |

---

## ⚡ Workflows

### 1. The "Hello World" of Forecasting

#### Query - Run forecasting on a demo dataset.

#### Response - Standard workflow followed by SKtime MCP for forecasting on a demo dataset.

**Step 1: Discover Data**
```json
{"tool": "list_available_data", "arguments": {"is_demo": true}}
```

**Step 2: Find a Forecaster**
```json
{"tool": "list_estimators", "arguments": {"task": "forecasting", "limit": 5}}
```

**Step 3: Instantiate the Estimator**
```json
{
  "tool": "instantiate_estimator",
  "arguments": {
    "estimator": "NaiveForecaster"
  }
}
```
Returns a handle, e.g. `{"success": true, "handle": "est_abc123", ...}`.  
Use the returned `handle` value in the next step as `estimator_handle`.

**Step 4: Fit & Predict**
```json
{
  "tool": "fit_predict",
  "arguments": {
    "estimator_handle": "est_abc123",
    "dataset": "airline",
    "horizon": 12
  }
}
```

### 2. Advanced Pipeline Composition

#### Query - Create sophisticated pipelines without writing complex code.

#### Response - Standard workflow followed by SKtime MCP. 

**Step 1: Validate Pipeline**
Check if components work together (e.g., Deseasonalizer -> Detrender -> ARIMA).
```json
{
  "tool": "validate_pipeline",
  "arguments": {"components": ["ConditionalDeseasonalizer", "Detrender", "ARIMA"]}
}
```

**Step 2: Instantiate Pipeline**
```json
{
  "tool": "instantiate_pipeline",
  "arguments": {
    "components": ["ConditionalDeseasonalizer", "Detrender", "ARIMA"],
    "params_list": [{}, {}, {"order": [1, 1, 1]}]
  }
}
```

### 3. Save a Fitted Model

Persist a trained estimator to a local filesystem path using sktime's MLflow integration.

**Step 1: Fit the estimator**
```json
{
  "tool": "fit_predict",
  "arguments": {
    "estimator_handle": "est_abc123",
    "dataset": "airline",
    "horizon": 12
  }
}
```

**Step 2: Save the fitted model**
```json
{
  "tool": "save_model",
  "arguments": {
    "estimator_handle": "est_abc123",
    "path": "/absolute/path/to/model_dir",
    "mlflow_params": {
      "serialization_format": "cloudpickle"
    }
  }
}
```

**Typical response**
```json
{
  "success": true,
  "estimator_handle": "est_abc123",
  "saved_path": "/absolute/path/to/model_dir",
  "message": "Model saved successfully to '/absolute/path/to/model_dir'"
}
```

---

## 💾 Data Management

Bring your own data into the MCP server.

### Supported Sources

*   **Local Files**: CSV, Parquet, Excel
*   **SQL Databases**: PostgreSQL, SQLite, etc.

### Example: Loading a CSV File

```json
{
  "tool": "load_data_source",
  "arguments": {
    "config": {
      "type": "file",
      "path": "/absolute/path/to/your/data.csv",
      "time_column": "timestamp",
      "target_column": "value"
    }
  }
}
```

!!! warning "Absolute Paths Required"
    The server requires **absolute file paths** (e.g., `/home/user/data.csv`). Relative paths may fail depending on where the server was started.

---

## 💡 Best Practices

- **Resource Management**: Explicitly release handles (`release_handle`, `release_data_handle`) when done to free up memory.
- **Reproducibility**: Always use `export_code` after a successful experiment to save your work.
- **Persistence**: Use `save_model` after fitting if you need the estimator to survive server restarts.
- **Data Hygiene**: Use `auto_format_on_load` for messy real-world data to avoid frequent validation errors.

---

## ⚠️ Known Limitations

While `sktime-mcp` is a powerful tool for prototyping, please be aware of the current architectural limitations.

#### 1. In-Memory Handles (Explicit Persistence Required)
The server stores active handles in standard Python dictionaries.
> **Impact**: If the server restarts or connection drops, in-memory handles are lost. Use `save_model` to persist fitted estimators to a local filesystem path when needed.

#### 2. Mixed Sync/Async Execution
Heavy operations can still block when using synchronous tools (like `fit_predict`).
> **Impact**: For long-running jobs, use async tools (`fit_predict_async`, `load_data_source_async`) so the server remains responsive.

#### 3. "The Data Wall" (Memory Limits)
Data Adapters read the entire dataset into RAM.
> **Impact**: Loading multi-gigabyte files may crash the server with an `OutOfMemory` error. Lazy loading is not yet supported.

#### 4. Security
Instantiation allows arbitrary parameters within the registry.
> **Impact**: While constrained to valid estimators, there is limited validation on parameter values, which could theoretically be misused.

#### 5. Rigid Data Formatting
The `auto_format` logic is heuristic-based.
> **Impact**: Complex time-series with irregular gaps or mixed frequencies might fail to auto-format correctly, requiring manual pre-processing outside the tool.

#### 6. Local-Only Filesystem
> **Impact**: The server cannot easily access files if running in an isolated Docker container unless volumes are mounted. It does not support "Upload over HTTP/MCP".

#### 7. JSON Serialization Loss
Complex sktime types (Periods, Intervals) are converted to strings for LLM consumption.
> **Impact**: Some rich metadata is lost during the conversion to JSON for the client response.

#### 8. Code Export Limitations
`export_code` uses template-based generation.
> **Impact**: Highly complex custom pipelines with lambda functions or specific edge-cases might generate code that requires minor manual fixes.

---

## ❓ Troubleshooting

| Issue | Solution |
|-------|----------|
| **"Unknown estimator"** | Use `search_estimators` to find the exact case-sensitive name. |
| **`No module named 'sktime'` / `sktime must be installed`** | Activate your project virtual environment and reinstall: `pip install -e ".[dev]"` (or `pip install -e ".[all]"` if you need optional adapters). |
| **"Missing dependencies"** | Run `pip install -e ".[all]"` to ensure optional extras are present. |
| **`save_model` import/runtime errors** | Install MLflow in the environment used by the server. The tool relies on `sktime.utils.mlflow_sktime.save_model` and saves to a local filesystem path. |
| **Validation Failures** | Enable `auto_format_on_load` or use `format_time_series` to clean your data. |
| **Server Timeout** | Heavy models take time. Be patient or try a simpler model (e.g., `NaiveForecaster`) first. |
