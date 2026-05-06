# 📘 User Guide

Welcome to the **sktime-mcp** User Guide. This guide walks you through installing, configuring, and using the MCP server for time-series forecasting — all through natural-language conversations with your AI assistant.

---

## 🚀 Getting Started

### Prerequisites

Before you begin, ensure you have:

- **Python 3.10+** installed.
- A compatible MCP client (like **Claude Desktop**, **Cursor**, or **VS Code with Copilot**).

### Installation

**Zero-install via uvx (recommended):** if you have [uv](https://github.com/astral-sh/uv) installed, no install step is needed. Just configure your MCP client (see below) and `uvx` handles everything automatically.

```bash
# Or install with pip
pip install sktime-mcp

# With all optional extras (SQL, forecasting models, file formats)
pip install "sktime-mcp[all]"
```

### MCP Client Configuration

**With uvx (recommended — no prior install needed):**
```json
{
  "mcpServers": {
    "sktime": {
      "command": "uvx",
      "args": ["sktime-mcp"]
    }
  }
}
```

**With optional extras:**
```json
{
  "mcpServers": {
    "sktime": {
      "command": "uvx",
      "args": ["sktime-mcp[forecasting,sql]"]
    }
  }
}
```

**With pip-installed package:**
```json
{
  "mcpServers": {
    "sktime": {
      "command": "sktime-mcp"
    }
  }
}
```

### Running the Server manually

```bash
sktime-mcp

# Or via Python
python -m sktime_mcp.server
```

!!! tip "Client Configuration"
    Ensure your MCP client (e.g., Claude Desktop) is configured to run this command. See the [official VSCode guidelines](https://code.visualstudio.com/docs/copilot/customization/mcp-servers#_configure-the-mcpjson-file) for configuration examples.

---

## How It Works

You interact with `sktime-mcp` through your AI assistant — you type requests in plain English, and the assistant translates them into the right sequence of operations behind the scenes. You never need to write JSON payloads or call tools directly.

The typical flow looks like this:

1. **You ask** your AI assistant a question or give it a task (e.g., *"Forecast airline passengers for the next 12 months"*).
2. **The assistant** selects the appropriate sktime-mcp tools, loads data, picks models, and runs them.
3. **You receive** results — forecasts, evaluation metrics, code, or summaries — in the conversation.

The sections below describe what you can ask for and what to expect.

---

## 🛠️ Core Capabilities

The `sktime-mcp` server exposes a suite of tools that your AI assistant uses on your behalf:

| What you can ask for | What happens behind the scenes | Example prompt |
|----------------------|-------------------------------|----------------|
| **Find models** | The assistant searches the sktime registry by task, tags, or keywords. | *"What forecasting models are available?"* |
| **Create a model or pipeline** | An estimator or multi-step pipeline is instantiated with your chosen parameters. | *"Set up an ARIMA(1,1,1) model"* |
| **Run a forecast** | The model is fitted on your data and predictions are generated. | *"Forecast the airline dataset 12 months ahead"* |
| **Cross-validate a model** | The model is evaluated across multiple folds using an expanding window to get metrics like MAE and RMSE. | *"Evaluate ARIMA using 3-fold cross-validation"* |
| **Run async background jobs** | Heavy training operations run in the background without blocking the client. | *"Fit this model in the background"* |
| **Load your own data** | CSV, Parquet, Excel, or SQL data is loaded and prepared for modelling. | *"Load my sales data from /home/user/sales.csv"* |
| **Export code** | A standalone Python script is generated so you can reproduce results outside the MCP server. | *"Give me the Python code for this model"* |
| **Save a trained model** | The fitted estimator is persisted to disk for later reuse. | *"Save this model to /home/user/models/arima"* |

---

## ⚡ Workflows

### 1. The "Hello World" of Forecasting

**What you want:** Run a forecast on a built-in demo dataset.

**Step 1: Find out what demo data is available**
Ask your assistant something like:

> *"What demo datasets can I use?"*

The assistant will return a list of built-in datasets (e.g., `airline`, `shampoo_sales`, `longley`). Each entry includes a short description and the shape of the data.

**Step 2: Pick a forecasting model**
Ask your assistant to suggest models:

> *"What forecasting models are available? Show me 5 options."*

The assistant will return a list of registered forecasters from sktime with their names and brief descriptions. You can ask for more detail on any of them (e.g., *"Tell me more about ARIMA"*).

**Step 3: Run the forecast**
Once you have picked a dataset and model, ask the assistant to run it:

> *"Forecast the airline dataset 12 months ahead using NaiveForecaster."*

The assistant will:

1. Instantiate the model you chose.
2. Fit it on the dataset.
3. Generate predictions for the requested horizon.
4. Return the forecast values to you.

You can then ask follow-up questions like *"Plot these results"*, *"Try a different model"*, or *"Give me the Python code for this"*.

<details>
<summary>🔧 Detailed MCP Tool Calls</summary>

The assistant translates this workflow into the following tool calls:

```json
{"tool": "list_available_data", "arguments": {"is_demo": true}}
```
```json
{"tool": "list_estimators", "arguments": {"task": "forecasting", "limit": 5}}
```
```json
{"tool": "instantiate_estimator", "arguments": {"estimator": "NaiveForecaster"}}
```
Returns a handle, e.g., `{"success": true, "handle": "est_abc123"}`.
```json
{"tool": "fit_predict", "arguments": {"estimator_handle": "est_abc123", "dataset": "airline", "horizon": 12}}
```

</details>

---

### 2. Forecasting on Your Own Data

**What you want:** Load a CSV file and forecast it.

**Step 1: Load your data**
Tell the assistant where your file is and which columns matter:

> *"Load my time series from /home/user/data/sales.csv. The time column is 'date' and the target column is 'revenue'."*

The assistant loads the file and reports back with a summary (number of rows, date range, detected frequency). If the data has formatting issues, the assistant will auto-format it or ask you how to handle problems.

**Step 2: Run a forecast**
Now ask for a forecast just like with demo data:

> *"Forecast revenue 6 months ahead using ARIMA."*

The assistant fits the model on your loaded data and returns predictions.

**Step 3 (optional): Export the code**
If you want a standalone Python script:

> *"Export the Python code for this model and data loading."*

You receive a script you can run independently of the MCP server.

<details>
<summary>🔧 Detailed MCP Tool Calls</summary>

```json
{
  "tool": "load_data_source",
  "arguments": {
    "config": {
      "type": "file",
      "path": "/home/user/data/sales.csv",
      "time_column": "date",
      "target_column": "revenue"
    }
  }
}
```
Returns a data handle, e.g., `{"handle": "data_xyz"}`.
```json
{"tool": "instantiate_estimator", "arguments": {"estimator": "ARIMA", "params": {"order": [1, 1, 1]}}}
```
```json
{"tool": "fit_predict", "arguments": {"estimator_handle": "est_abc123", "data_handle": "data_xyz", "horizon": 6}}
```
```json
{"tool": "export_code", "arguments": {"handle": "est_abc123", "include_fit_example": true}}
```

</details>

!!! warning "Absolute Paths Required"
    The server requires **absolute file paths** (e.g., `/home/user/data.csv`). Relative paths may fail depending on where the server was started.

---

### 3. Advanced Pipeline Composition

**What you want:** Build a multi-step preprocessing + forecasting pipeline.

**Step 1: Describe the pipeline you want**
Tell the assistant the components and their order:

> *"Build a pipeline that deseasonalizes, detrends, then applies ARIMA(1,1,1) to forecast."*

The assistant will first validate that these components are compatible (e.g., that transformers feed into a forecaster correctly). If something is incompatible, it will explain why and suggest alternatives.

**Step 2: Run it on data**
Once the pipeline is created, use it like any other model:

> *"Run this pipeline on the airline dataset for 12 months ahead."*

The entire pipeline (preprocessing + forecasting) runs end-to-end, and you get predictions back.

<details>
<summary>🔧 Detailed MCP Tool Calls</summary>

```json
{
  "tool": "validate_pipeline",
  "arguments": {"components": ["ConditionalDeseasonalizer", "Detrender", "ARIMA"]}
}
```
```json
{
  "tool": "instantiate_pipeline",
  "arguments": {
    "components": ["ConditionalDeseasonalizer", "Detrender", "ARIMA"],
    "params_list": [{}, {}, {"order": [1, 1, 1]}]
  }
}
```
```json
{
  "tool": "fit_predict",
  "arguments": {"estimator_handle": "est_xyz789", "dataset": "airline", "horizon": 12}
}
```

</details>

---

### 4. Saving a Trained Model

**What you want:** Persist a fitted model so it survives server restarts.

After fitting a model (in any of the workflows above), ask:

> *"Save this model to /home/user/models/my_forecaster."*

The assistant persists the fitted estimator to the specified path using sktime's MLflow integration. You can later reload it:

> *"Load the model from /home/user/models/my_forecaster and predict 6 months ahead on the airline dataset."*

<details>
<summary>🔧 Detailed MCP Tool Calls</summary>

**Saving:**
```json
{
  "tool": "save_model",
  "arguments": {
    "estimator_handle": "est_abc123",
    "path": "/home/user/models/my_forecaster"
  }
}
```

**Loading and predicting:**
```json
{
  "tool": "load_model",
  "arguments": {"path": "/home/user/models/my_forecaster"}
}
```
```json
{
  "tool": "fit_predict",
  "arguments": {"estimator_handle": "est_restored", "dataset": "airline", "horizon": 6}
}
```

</details>

---

## 💾 Data Management

### Supported Data Sources

| Source | How to use it | Example prompt |
|--------|--------------|----------------|
| **Demo datasets** | Reference by name — no file paths needed. | *"Use the airline dataset"* |
| **Local CSV / Parquet / Excel** | Provide the absolute path, time column, and target column. | *"Load /home/user/data.csv with time column 'date' and target 'value'"* |
| **SQL databases** | Provide connection details, query, and column names. | *"Load data from my PostgreSQL database using this query: SELECT ..."* |

For full details on data source configuration, see the [Data Sources](data-sources.md) page.

### Tips for Loading Data

- **Always use absolute paths** for local files.
- If your data has formatting issues (missing values, duplicate timestamps, irregular frequency), mention it and the assistant can auto-format it for you.
- After loading, you can ask *"Describe my loaded data"* to see a summary before modelling.

---

## 💡 Best Practices

- **Start with demo data** to familiarize yourself with the workflow before loading your own files.
- **Ask for code export** after a successful experiment — this gives you a reproducible Python script.
- **Save fitted models** if you need them to survive server restarts.
- **Clean up when done** — ask the assistant to release data and model handles to free memory (e.g., *"Release all handles"*).
- **Use background execution for heavy jobs** — if a model takes a long time to fit, mention that you want it to run in the background (e.g., *"Fit this in the background and let me know when it's done"*). See [Background Jobs](background-jobs.md) for details.

---

## ⚠️ Known Limitations

While `sktime-mcp` is a powerful tool for prototyping, please be aware of the current limitations.

#### 1. In-Memory Handles (Explicit Persistence Required)
The server stores active handles in standard Python dictionaries.
> **Impact**: If the server restarts or connection drops, in-memory handles are lost. Use `save_model` to persist fitted estimators when needed.

#### 2. Mixed Sync/Async Execution
Heavy operations can block when using synchronous tools.
> **Impact**: For long-running jobs, ask the assistant to run them in the background so the server remains responsive.

#### 3. Memory Limits
Data is read entirely into RAM.
> **Impact**: Loading multi-gigabyte files may crash the server. Consider pre-filtering large datasets before loading.

#### 4. Security
Instantiation allows arbitrary parameters within the registry.
> **Impact**: While constrained to valid estimators, there is limited validation on parameter values.

#### 5. Rigid Data Formatting
The auto-format logic is heuristic-based.
> **Impact**: Complex time series with irregular gaps or mixed frequencies might fail to auto-format correctly, requiring manual pre-processing outside the tool.

#### 6. Local-Only Filesystem
> **Impact**: The server cannot access files inside isolated Docker containers unless volumes are mounted. Direct HTTP upload is not yet supported.

#### 7. JSON Serialization Loss
Complex sktime types (Periods, Intervals) are converted to strings.
> **Impact**: Some rich metadata is simplified when returned in conversation.

#### 8. Code Export Limitations
`export_code` uses template-based generation.
> **Impact**: Highly complex pipelines with lambda functions or edge cases might generate code that needs minor manual fixes.

---

## ❓ Troubleshooting

| Problem | What to do |
|---------|------------|
| **"Unknown estimator"** | Ask the assistant to search for the model by name — the exact casing matters. |
| **`No module named 'sktime'`** | Activate your project virtual environment and reinstall: `pip install -e ".[dev]"`. |
| **"Missing dependencies"** | Run `pip install -e ".[all]"` to ensure all optional extras are present. |
| **Model saving fails** | Ensure MLflow is installed in the server's environment. |
| **Data validation failures** | Ask the assistant to auto-format the data, or pre-process your file to fix timestamp issues. |
| **Server seems unresponsive** | A heavy model may be fitting synchronously. Wait for it to complete, or use background execution next time. |
