# sktime-mcp

**[Read the Documentation](http://sktime.github.io/sktime-mcp/)** | **[PyPI Package](https://pypi.org/project/sktime-mcp/0.1.0/)**


**MCP (Model Context Protocol) layer for sktime - Registry-Driven for LLMs**

A semantic engine that exposes sktime's native registry and semantics to Large Language Models, enabling them to:

- 🔍 **Discover** valid estimators
- 🧠 **Reason** about estimator capabilities  
- 🔗 **Compose** compatible estimators
- ⚡ **Execute** real sktime workflows on real data

## 🎯 Design Philosophy

This MCP is **not** just documentation or static code analysis. It is a **semantic engine** for programmatic model usage.

### Key Principles

1. **sktime as Source of Truth** - No AST parsing, no repo indexing, no heuristics. All structure comes from `all_estimators`, estimator tags, and sktime's API contracts.

2. **Registry-First** - Instead of `File → Class → Infer Relationships`, we do `Registry → Semantics → Safe Execution`.

3. **Minimal MCP Surface** - Exposes only what an LLM needs: Discovery, Description, Instantiation, Execution, and model persistence.

## 🛠️ Installation

### Zero-install via uvx (recommended)

If you have [uv](https://github.com/astral-sh/uv) installed, no separate installation step is needed. Just update your MCP client config (see [Connecting from an LLM Client](#connecting-from-an-llm-client) below) and `uvx` will handle the rest automatically.

```bash
# Verify uv is available
uvx sktime-mcp --help
```

### pip

```bash
pip install sktime-mcp

# With optional extras (SQL, forecasting models, file formats)
pip install "sktime-mcp[all]"
```

### Development installation

```bash
git clone https://github.com/sktime/sktime-mcp
cd sktime-mcp
python3 -m pip install -e ".[dev]"
```
## 🧭 Beginner Setup (First‑Time Users)

If you are new to sktime‑mcp or to MCP‑based workflows, this section provides a minimal starting point to help you verify that your setup is working correctly.

### What is MCP?
The Model Context Protocol (MCP) allows Large Language Models (LLMs) to discover, reason about, and execute sktime workflows programmatically. This project exposes sktime’s estimator registry and semantics in a structured way so that LLMs can safely compose and run real time‑series pipelines.

### Prerequisites
- Python 3.10 or newer
- A working Python virtual environment (recommended)
- `pip` installed

### macOS / Unix-like first-time setup

For macOS or Unix-like shells, create an isolated virtual environment before installing the package:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install sktime-mcp
```

For development (if you want to modify the source):

```bash
python -m pip install -e ".[dev]"
```

**Verify that the MCP server starts:**

```bash
sktime-mcp
```

If the `sktime-mcp` console command is not found (e.g. the script was not placed on your `PATH`), use the module fallback instead — this is also the recommended form when an MCP client needs to target a specific Python environment:

```bash
python -m sktime_mcp.server
```

**Common first-time issues:**

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `command not found: sktime-mcp` | Scripts directory not on `PATH` | Run `python -m sktime_mcp.server` or add `.venv/bin` to your `PATH` |
| `ModuleNotFoundError: sktime_mcp` | Package not installed in the active environment | Confirm `.venv` is active (`which python`) and re-run `pip install sktime-mcp` |
| `pip: command not found` | System `pip` not available | Use `python -m pip` instead of bare `pip` |
| Wrong Python version selected | Multiple Python installations | Invoke `python3 -m venv .venv` explicitly and always use `python` inside the activated environment |

### Minimal Setup Check

After completing the steps above, confirm the server starts with `sktime-mcp`. See the [macOS / Unix-like first-time setup](#macos--unix-like-first-time-setup) section for the fallback command and common error solutions.

> **Note:** On Windows, the `sktime-mcp` command may be installed to a directory
> not on your `PATH` (e.g., `%APPDATA%\Python\Python3xx\Scripts`). Either add
> that directory to your `PATH` or use `python -m sktime_mcp.server` instead.


## 🚀 Quick Start

### Running the MCP Server
#### Standard Stdio Mode (for MCP Clients)
```bash
sktime-mcp
```

#### HTTP/SSE Mode via FastAPI (for Web Browsers or ChatGPT)
To expose the MCP server as a REST API over SSE (Server-Sent Events) for direct consumption:
```bash
PYTHONPATH=src .venv/bin/uvicorn sktime_mcp.app:app --host 127.0.0.1 --port 8001
```
This exposes standard SSE on `/sse` and message passing on `/messages/`.

> **Note for ChatGPT Web Users:** ChatGPT runs in the cloud and cannot connect to `http://127.0.0.1` (you will get an "Unsafe URL" error). You must expose your local server to the internet using a secure tunnel like [ngrok](https://ngrok.com/):
> ```bash
> ngrok http 8001
> ```
> Then use the provided `https://<your-ngrok-id>.ngrok-free.app/sse` URL in ChatGPT.

### Configuration (Environment Variables)

You can configure the server's behavior at runtime using environment variables:

- `SKTIME_MCP_MAX_RESPONSE_TOKENS`: Maximum tokens allowed per tool response (e.g., `10000`). If a response exceeds this limit, it is truncated and appended with a notice. Set to `0` (default) for unlimited.
- `SKTIME_MCP_LOG_LEVEL`: Server logging verbosity level (`DEBUG`, `INFO`, `WARNING`, `ERROR`). Defaults to `WARNING`.
- `SKTIME_MCP_AUTO_FORMAT`: Enables or disables automatic time-series formatting during data loading.
- `SKTIME_MCP_JOB_MAX_AGE_HOURS`: Maximum hours before completed background jobs are automatically pruned. Defaults to `24`.

### Connecting from an LLM Client

The server uses stdio transport by default, compatible with Claude Desktop, Claude Code, and other MCP clients.

**Claude Desktop** — add to your config file:

| Platform | Config path |
|----------|-------------|
| macOS    | `~/Library/Application Support/Claude/claude_desktop_config.json` |
| Linux    | `~/.config/claude/claude_desktop_config.json` |
| Windows  | `%APPDATA%\Claude\claude_desktop_config.json` |

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

## ⚙️ Configuration

The server can be configured via environment variables:

| Environment Variable | Description | Default |
|----------------------|-------------|---------|
| `SKTIME_MCP_LOG_LEVEL` | Logging verbosity (e.g. `INFO`, `DEBUG`, `WARNING`) | `"WARNING"` |
| `SKTIME_MCP_LOG_PATH` | Optional file path to output logs to in addition to stderr | (None) |
| `SKTIME_MCP_AUTO_FORMAT` | Automatically format time series data on load (`true`/`false`) | `"true"` |
| `SKTIME_MCP_JOB_MAX_AGE_HOURS` | Maximum age in hours before background jobs are cleared | `24` |
| `SKTIME_MCP_JOB_CLEANUP_INTERVAL` | Interval in seconds for periodic job cleanup checks | `3600` |

## 📚 Available Tools

The `sktime-mcp` server exposes a rich suite of tools categorized logically. Every tool is fully typed, specifies clear return schemas, and aligns with native `sktime` naming conventions wherever possible.

---

### Category 1: Discovery & Registry Tools

These tools enable the LLM to inspect the native `sktime` registry, search for estimators matching specific criteria, and understand their capability profiles.

#### 1. `list_estimators`
Discover estimator classes from the `sktime` registry by task type, capabilities, or name query.
* **Arguments:**
  * `task` (`str`, optional): Filter by task type. Valid values: `"forecasting"`, `"classification"`, `"regression"`, `"transformation"`, `"clustering"`, `"detection"`.
  * `tags` (`dict[str, Any]`, optional): Key-value pairs filtering by capability tags (e.g., `{"capability:pred_int": true, "handles-missing-data": true}`).
  * `query` (`str`, optional): Case-insensitive substring search on the estimator's class name or docstring.
  * `limit` (`int`, optional, default=`50`): Maximum number of results to return.
  * `offset` (`int`, optional, default=`0`): Number of entries to skip for pagination.
* **Returns:**
  ```json
  {
    "success": true,
    "estimators": [
      {
        "name": "ARIMA",
        "task": "forecasting",
        "description": "Autoregressive Integrated Moving Average (ARIMA) model...",
        "tags": {
          "capability:pred_int": true,
          "handles-missing-data": false
        },
        "import_path": "sktime.forecasting.arima.ARIMA"
      }
    ],
    "total": 1
  }
  ```

#### 2. `describe_estimator`
Get full documentation, hyperparameters, capability tags, and Python import path for a specific named estimator class.
* **Arguments:**
  * `estimator` (`str`, required): Name of the estimator class (e.g., `"ARIMA"`, `"NaiveForecaster"`).
* **Returns:**
  ```json
  {
    "success": true,
    "name": "ARIMA",
    "description": "Autoregressive Integrated Moving Average...",
    "params": {
      "order": {
        "type": "tuple",
        "default": [1, 0, 0],
        "description": "The (p,d,q) order of the model."
      }
    },
    "tags": {
      "capability:pred_int": true
    },
    "import_path": "sktime.forecasting.arima.ARIMA"
  }
  ```

#### 3. `get_available_tags`
List all queryable capability tags across the `sktime` registry with their descriptions and expected value types.
* **Arguments:** None.
* **Returns:**
  ```json
  {
    "success": true,
    "tags": {
      "capability:pred_int": {
        "type": "bool",
        "description": "Whether the forecaster can compute prediction intervals.",
        "scitype": "forecaster"
      }
    }
  }
  ```

---

### Category 2: Instantiation & Composition Tools

These tools manage the creation and lifecycle of model instances in server memory.

#### 4. `instantiate`
Unify single estimators and multi-step composite pipelines under a single instantiator, returning an active `estimator_handle`.
* **Arguments:**
  * `estimator` (`str | list[str]`, required): Either a single estimator class name or an ordered list of class names representing a pipeline (e.g., `["Detrender", "ARIMA"]`).
  * `params` (`dict[str, Any] | list[dict[str, Any]]`, optional): Hyperparameters for the single estimator, or a list of parameter dicts corresponding element-wise to the pipeline steps.
* **Returns:**
  ```json
  {
    "success": true,
    "estimator_handle": "est_abc123",
    "structure": "Detrender ➔ ARIMA",
    "params": [{}, {"order": [1, 1, 1]}]
  }
  ```

#### 5. `validate_pipeline`
Verify the compatibility and validity of a list of estimator steps in pipeline order before instantiation.
* **Arguments:**
  * `components` (`list[str]`, required): List of estimator class names in pipeline order.
* **Returns:**
  ```json
  {
    "success": true,
    "valid": true,
    "errors": [],
    "warnings": []
  }
  ```

#### 6. `list_handles`
List all active estimator and pipeline handles currently allocated in server memory.
* **Arguments:** None.
* **Returns:**
  ```json
  {
    "success": true,
    "handles": [
      {
        "handle": "est_abc123",
        "class_name": "Pipeline",
        "is_fitted": false,
        "created_at": "2026-05-19T10:30:00Z"
      }
    ]
  }
  ```

#### 7. `release_handle`
Free an estimator or pipeline handle from server memory to prevent leaks.
* **Arguments:**
  * `handle` (`str`, required): The handle ID to release.
* **Returns:**
  ```json
  {
    "success": true,
    "message": "Handle 'est_abc123' released successfully."
  }
  ```

---

### Category 3: Execution & Application Tools

Execution tools apply estimator handles to data. Rather than segregating custom files and built-in demos, arguments like `y` and `X` accept **either** a registered `data_handle` or a built-in demo dataset name (e.g., `"airline"`).

#### 8. `fit` / `fit_async`
Fit an instantiated estimator or pipeline on a target time series `y` and optional exogenous data `X`.
* **Arguments:**
  * `estimator_handle` (`str`, required): Active estimator handle.
  * `y` (`str`, required): Target time series. Can be an active `data_handle` or a built-in demo dataset name (e.g., `"airline"`).
  * `X` (`str`, optional): Exogenous time series (active `data_handle` or demo dataset name).
  * `fit_params` (`dict[str, Any]`, optional): Extra parameters forwarded to the underlying `fit` call (e.g., `fh` if required at fit time).
* **Returns (`fit`):**
  ```json
  {
    "success": true,
    "estimator_handle": "est_abc123",
    "message": "Estimator fitted successfully."
  }
  ```
* **Returns (`fit_async`):**
  ```json
  {
    "success": true,
    "job_id": "job_fit_xyz789",
    "message": "Fitting job started in background."
  }
  ```

#### 9. `predict` / `predict_async`
Generate predictions from a fitted estimator or pipeline.
* **Arguments:**
  * `estimator_handle` (`str`, required): Fitted estimator handle.
  * `X` (`str`, optional): Exogenous/feature series (active `data_handle` or demo name).
  * `fh` (`list[int] | list[str] | int`, optional): Forecast horizon (for forecasters).
  * `predict_params` (`dict[str, Any]`, optional): Extra parameters forwarded to `predict`.
* **Returns (`predict`):**
  ```json
  {
    "success": true,
    "predictions": [
      {"index": "1961-01", "value": 450.2},
      {"index": "1961-02", "value": 455.1}
    ],
    "fh": [1, 2]
  }
  ```
* **Returns (`predict_async`):**
  ```json
  {
    "success": true,
    "job_id": "job_pred_123456",
    "message": "Prediction job started in background."
  }
  ```

#### 10. `predict_interval`
Generate prediction intervals from a fitted probabilistic forecaster.
* **Arguments:**
  * `estimator_handle` (`str`, required): Fitted estimator handle.
  * `X` (`str`, optional): Exogenous/feature series.
  * `fh` (`list[int] | list[str] | int`, optional): Forecast horizon.
  * `coverage` (`list[float] | float`, optional, default=`0.9`): Nominal coverage interval(s) in `(0, 1)`.
* **Returns:**
  ```json
  {
    "success": true,
    "intervals": [
      {
        "index": "1961-01",
        "lower_0.9": 435.1,
        "upper_0.9": 465.3
      }
    ],
    "coverage": [0.9]
  }
  ```

#### 11. `predict_quantiles`
Generate quantile forecasts from a fitted probabilistic forecaster.
* **Arguments:**
  * `estimator_handle` (`str`, required): Fitted estimator handle.
  * `X` (`str`, optional): Exogenous series.
  * `fh` (`list[int] | list[str] | int`, optional): Forecast horizon.
  * `alpha` (`list[float] | float`, required): Quantile values to compute in `(0, 1)`.
* **Returns:**
  ```json
  {
    "success": true,
    "quantiles": [
      {
        "index": "1961-01",
        "quantile_0.05": 420.3,
        "quantile_0.95": 480.1
      }
    ],
    "alpha": [0.05, 0.95]
  }
  ```

#### 12. `transform`
Apply a transformer to data, registering the transformed output as a new `data_handle`.
* **Arguments:**
  * `estimator_handle` (`str`, required): Instantiated transformer handle.
  * `X` (`str`, required): Series/DataFrame to transform (active `data_handle` or demo name).
  * `y` (`str`, optional): Target series (active `data_handle` or demo name).
  * `action` (`str`, optional, default=`"transform"`): Action to perform. Valid values: `"transform"`, `"fit_transform"`, `"inverse_transform"`.
* **Returns:**
  ```json
  {
    "success": true,
    "data_handle": "data_trans_999",
    "metadata": {
      "shape": [144, 1],
      "frequency": "M",
      "columns": ["value_transformed"]
    }
  }
  ```

#### 13. `update`
Update a fitted forecaster/model with new incoming streaming data (online learning).
* **Arguments:**
  * `estimator_handle` (`str`, required): Fitted estimator handle.
  * `y` (`str`, required): New target observations (active `data_handle` or demo name).
  * `X` (`str`, optional): New exogenous observations (active `data_handle` or demo name).
  * `update_params` (`dict[str, Any]`, optional): Extra parameters passed to the `update` call.
* **Returns:**
  ```json
  {
    "success": true,
    "message": "Model updated successfully with 12 new observations."
  }
  ```

#### 14. `get_fitted_params`
Retrieve learnable coefficients, components, or parameters from a fitted estimator.
* **Arguments:**
  * `estimator_handle` (`str`, required): Fitted estimator handle.
* **Returns:**
  ```json
  {
    "success": true,
    "fitted_params": {
      "cutoff": "1960-12",
      "sigma2": 0.0431,
      "ar_coefficients": [0.824]
    }
  }
  ```

---

### Category 4: Evaluation Tools

These tools streamline model validation and scoring pipelines.

#### 15. `evaluate` / `evaluate_async`
Cross-validate an estimator on a dataset to assess performance.
* **Arguments:**
  * `estimator_handle` (`str`, required): Instantiated estimator handle.
  * `y` (`str`, required): Target time series (active `data_handle` or demo name).
  * `X` (`str`, optional): Exogenous time series.
  * `cv_folds` (`int`, optional, default=`3`): Number of folds for cross-validation.
  * `metric` (`str`, optional): Performance metric name (e.g., `"MeanAbsolutePercentageError"`).
* **Returns (`evaluate`):**
  ```json
  {
    "success": true,
    "metrics": {
      "mean_absolute_percentage_error": 0.054
    },
    "fold_results": [
      {"fold": 0, "score": 0.049},
      {"fold": 1, "score": 0.058}
    ]
  }
  ```
* **Returns (`evaluate_async`):**
  ```json
  {
    "success": true,
    "job_id": "job_eval_111",
    "message": "Evaluation job started in background."
  }
  ```

---

### Category 5: Data Management Tools

These tools manage input and output datasets, providing caching, standardization, and memory reclamation.

#### 16. `list_available_data`
List all built-in demo datasets and active user-loaded data handles in a single response.
* **Arguments:**
  * `is_demo` (`bool`, optional): Filter to built-in system demos (`true`) or active in-memory handles (`false`). Omit to get both.
* **Returns:**
  ```json
  {
    "success": true,
    "system_demos": ["airline", "sunspots", "lynx"],
    "active_handles": [
      {
        "data_handle": "data_abc123",
        "shape": [144, 2],
        "columns": ["target", "exogenous"],
        "frequency": "M"
      }
    ]
  }
  ```

#### 17. `load_data_source` / `load_data_source_async`
Load custom files, SQL database queries, URLs, or inline JSON into the server as a `data_handle`.
* **Arguments:**
  * `config` (`dict[str, Any]`, required): Source configuration dictionary. Must contain:
    * `"type"` (`str`, required): `"pandas"` (inline dict), `"file"` (CSV/Excel/Parquet), `"sql"` (DB connection), or `"url"`.
    * Additional keys based on `"type"` (e.g., `"path"`, `"url"`, `"connection_string"`, `"query"`, `"time_column"`, `"target_column"`).
* **Returns (`load_data_source`):**
  ```json
  {
    "success": true,
    "data_handle": "data_abc123",
    "metadata": {
      "shape": [500, 1],
      "columns": ["sales"],
      "frequency": "D"
    }
  }
  ```
* **Returns (`load_data_source_async`):**
  ```json
  {
    "success": true,
    "job_id": "job_load_444",
    "message": "Asynchronous loading started."
  }
  ```

#### 18. `save_data`
Persist an in-memory `data_handle` (such as predictions or transformed series) back to disk.
* **Arguments:**
  * `data_handle` (`str`, required): In-memory data handle to save.
  * `path` (`str`, required): Local filesystem path where the file will be saved.
  * `format` (`str`, optional): Output format (inferred from path extension if omitted: `"csv"`, `"parquet"`, `"json"`).
* **Returns:**
  ```json
  {
    "success": true,
    "saved_path": "/home/user/forecasts.csv",
    "message": "Data saved successfully."
  }
  ```

#### 19. `format_time_series`
Clean, fill missing values, deduplicate, and standardize loaded time series data.
* **Arguments:**
  * `data_handle` (`str`, required): Target data handle.
  * `auto_infer_freq` (`bool`, optional, default=`true`): Re-infer time delta frequency.
  * `fill_missing` (`bool`, optional, default=`true`): Interpolate missing values using forward/backward fills.
  * `remove_duplicates` (`bool`, optional, default=`true`): Deduplicate timestamps.
* **Returns:**
  ```json
  {
    "success": true,
    "data_handle": "data_abc123",
    "changes_applied": ["inferred frequency: M", "filled 3 missing values"]
  }
  ```

#### 20. `release_data_handle`
Free a data handle and its contents from server memory.
* **Arguments:**
  * `data_handle` (`str`, required): Handle ID to release.
* **Returns:**
  ```json
  {
    "success": true,
    "message": "Data handle 'data_abc123' freed from memory."
  }
  ```

---

### Category 6: Persistence & Code Generation

These tools manage the serialization of estimator instances and generation of production-ready source code.

#### 21. `save_model`
Serialize an estimator blueprint or fitted model handle to disk using sktime-MLflow integration.
* **Arguments:**
  * `estimator_handle` (`str`, required): Estimator or pipeline handle to save.
  * `path` (`str`, required): Filesystem path or MLflow artifact URI.
  * `mlflow_params` (`dict[str, Any]`, optional): Extra parameters for MLflow.
* **Returns:**
  ```json
  {
    "success": true,
    "saved_path": "/path/to/model_dir",
    "is_fitted": true,
    "message": "Fitted model saved successfully."
  }
  ```

#### 22. `load_model`
Reload a serialized blueprint or fitted model back into an active `estimator_handle`.
* **Arguments:**
  * `path` (`str`, required): Filesystem path to the model directory.
* **Returns:**
  ```json
  {
    "success": true,
    "estimator_handle": "est_rehydrated_999",
    "is_fitted": true
  }
  ```

#### 23. `export_code`
Generate standalone, executable Python code to reproduce an estimator's structure and execution.
* **Arguments:**
  * `handle` (`str`, required): Handle ID of the estimator/pipeline.
  * `var_name` (`str`, optional, default=`"model"`): Variable name in the generated code block.
  * `include_fit_example` (`bool`, optional, default=`false`): Whether to inject boilerplate loading/fitting logic.
  * `y` (`str`, optional): Dataset name for the fitting logic (default: `"airline"`).
* **Returns:**
  ```json
  {
    "success": true,
    "code": "from sktime.forecasting.arima import ARIMA

model = ARIMA(order=(1, 1, 1))
..."
  }
  ```

---

### Category 7: Background Jobs

These tools poll, track, and manage asynchronous task executions.

#### 24. `check_job_status`
Inspect the status, execution progress, and results of a background job.
* **Arguments:**
  * `job_id` (`str`, required): Unique ID of the job.
* **Returns:**
  ```json
  {
    "success": true,
    "job_id": "job_fit_xyz789",
    "status": "COMPLETED",
    "progress_pct": 100.0,
    "error": null,
    "result": {
      "estimator_handle": "est_fitted_123",
      "message": "Estimator fitted successfully."
    }
  }
  ```

#### 25. `list_jobs`
List all active and historical background jobs.
* **Arguments:**
  * `status` (`str`, optional): Filter by status (`"pending"`, `"running"`, `"completed"`, `"failed"`, `"cancelled"`).
  * `limit` (`int`, optional, default=`20`): Pagination limit.
* **Returns:**
  ```json
  {
    "success": true,
    "jobs": [
      {
        "job_id": "job_fit_xyz789",
        "status": "completed",
        "created_at": "2026-05-19T10:32:00Z"
      }
    ]
  }
  ```

#### 26. `cancel_job`
Prune or terminate an active background job.
* **Arguments:**
  * `job_id` (`str`, required): Unique ID of the job.
  * `delete` (`bool`, optional, default=`false`): If true, deletes the job tracking record from memory.
* **Returns:**
  ```json
  {
    "success": true,
    "message": "Job 'job_fit_xyz789' terminated."
  }
  ```

---

---

#### 18. `auto_format_on_load`
Enable or disable automatic time series formatting whenever data is loaded via `load_data_source`. Auto-formatting is **enabled by default** and handles frequency inference, duplicate removal, and missing value filling.

**Arguments:**
- `enabled` (required): `true` to enable auto-formatting, `false` to disable

**Example:**
```json
{"enabled": false}
```

**Returns:** `{"success": true, "auto_format_enabled": false}`

---

### Handle Management

#### 19. `list_handles`
List all active estimator handles currently loaded in server memory.

**Arguments:** None

**Returns:**
```json
{"success": true, "handles": [{"handle": "est_abc123", "estimator": "ARIMA", "fitted": true}], "count": 1}
```

---

#### 20. `release_handle`
Release an estimator handle and free it from server memory. Call this after you are done with an estimator to avoid memory leaks in long-running sessions.

**Arguments:**
- `handle` (required): Handle ID to release

**Example:**
```json
{"handle": "est_abc123"}
```

**Returns:** `{"success": true, "message": "Handle est_abc123 released."}`

---

#### 21. `release_data_handle`
Release a loaded data handle and free its memory. The counterpart to `load_data_source`.

**Arguments:**
- `data_handle` (required): Data handle ID to release

**Example:**
```json
{"data_handle": "data_abc123"}
```

**Returns:** `{"success": true, "message": "Data handle data_abc123 released."}`

---

### Evaluation

#### 22. `evaluate_estimator`
Evaluate an estimator using expanding-window cross-validation. Returns per-fold metrics (e.g., MAPE, RMSE).

**Arguments:**
- `estimator_handle` (required): Handle from `instantiate_estimator`
- `dataset` (required): Dataset name (e.g., `"airline"`, `"sunspots"`, `"lynx"`)
- `cv_folds` (optional): Number of cross-validation folds (default: 3)

**Example:**
```json
{
  "estimator_handle": "est_abc123",
  "dataset": "airline",
  "cv_folds": 5
}
```

**Returns:** `{"success": true, "results": [{"test_MeanAbsolutePercentageError": 0.04, ...}, ...], "cv_folds_run": 5}`

---

### Additional Data Loading

#### 23. `load_data_source_async`
Non-blocking version of `load_data_source`. Schedules the data loading as a background job and returns a `job_id` immediately. The `data_handle` is available in the job result when the job reaches `completed` status.

**Arguments:**
- `config` (required): Same configuration object as `load_data_source`

**Example:**
```json
{
  "config": {
    "type": "file",
    "path": "/path/to/large_data.csv",
    "time_column": "date",
    "target_column": "sales"
  }
}
```

**Returns:** `{"success": true, "job_id": "abc-123", "message": "Data loading started..."}`

> Use `check_job_status(job_id)` to poll for completion and retrieve the `data_handle`.

---

#### 24. `list_data_sources`
List all registered data source adapter types and their descriptions. Call this to discover which `"type"` values are valid in `load_data_source` config.

**Arguments:** None

**Returns:**
```json
{
  "success": true,
  "sources": ["pandas", "file", "sql", "url"],
  "descriptions": {
    "pandas": {"class": "PandasAdapter", "description": "Load data from a pandas DataFrame"},
    "file":   {"class": "FileAdapter",   "description": "Load data from CSV, Parquet, or Excel files"}
  }
}
```

---

### Additional Background Jobs

#### 25. `cancel_job`
Cancel a running or pending background job (e.g., a long `fit_predict_async` or `load_data_source_async`).

**Arguments:**
- `job_id` (required): Job ID to cancel

**Example:**
```json
{"job_id": "abc-123"}
```

**Returns:** `{"success": true, "message": "Job abc-123 cancelled."}`

---

#### 26. `delete_job`
Permanently delete a job record from the job manager. The job will no longer appear in `list_jobs` results.

**Arguments:**
- `job_id` (required): Job ID to delete

**Example:**
```json
{"job_id": "abc-123"}
```

**Returns:** `{"success": true}`

---

#### 27. `cleanup_old_jobs`
Remove all completed, failed, or cancelled job records older than a given number of hours. Useful for keeping the job list clean in long-running sessions.

**Arguments:**
- `max_age_hours` (optional): Remove jobs older than this (default: 24)

**Example:**
```json
{"max_age_hours": 12}
```

**Returns:** `{"success": true, "deleted_count": 5}`

---

### Additional Model Persistence

#### 28. `load_model`
Load a previously saved sktime model from disk and register it as an active estimator handle, ready for `predict` or `fit_predict`.

**Arguments:**
- `path` (required): Path to the saved model directory (created by `save_model`)

**Example:**
```json
{"path": "/absolute/path/to/model_dir"}
```

**Returns:** `{"success": true, "handle": "est_abc123", "estimator": "ARIMA", "fitted": true}`

---

## 🔄 Example LLM Flows

Below are examples demonstrating how an LLM utilizes these redesigned tools to construct, validate, and execute workflows.

### Flow 1: Simple Probabilistic Forecasting

**User Prompt:** "Forecast monthly airline passengers using a probabilistic model."

**LLM Steps:**

1. **Discover Models**
   The LLM queries for forecasting models supporting prediction intervals.
   ```json
   // list_estimators
   {
     "task": "forecasting",
     "tags": {
       "capability:pred_int": true
     }
   }
   ```
   *Returns:* A list containing `"ARIMA"` and `"NaiveForecaster"`.

2. **Inspect Choice**
   The LLM inspects the parameter schema of `"ARIMA"`.
   ```json
   // describe_estimator
   {
     "estimator": "ARIMA"
   }
   ```

3. **Instantiate**
   The LLM instantiates the ARIMA model with specific parameters.
   ```json
   // instantiate
   {
     "estimator": "ARIMA",
     "params": {
       "order": [1, 1, 1]
     }
   }
   ```
   *Returns:* `{"success": true, "estimator_handle": "est_arima_456", ...}`

4. **Fit the Model**
   The LLM fits the model to the built-in `"airline"` demo dataset.
   ```json
   // fit
   {
     "estimator_handle": "est_arima_456",
     "y": "airline"
   }
   ```
   *Returns:* `{"success": true, "estimator_handle": "est_arima_456", ...}`

5. **Generate Forecast Intervals**
   The LLM computes a 90% confidence prediction interval for the next 12 months.
   ```json
   // predict_interval
   {
     "estimator_handle": "est_arima_456",
     "fh": 12,
     "coverage": 0.90
   }
   ```
   *Returns:* Predicton interval upper and lower bounds.

---

### Flow 2: Pipeline Composition & Forecasting

**User Prompt:** "Compose a forecasting pipeline that includes seasonal deseasonalization, detrending, and ARIMA."

**LLM Steps:**

1. **Validate Composition**
   The LLM validates compatibility of the steps before allocating handles.
   ```json
   // validate_pipeline
   {
     "components": ["ConditionalDeseasonalizer", "Detrender", "ARIMA"]
   }
   ```
   *Returns:* `{"success": true, "valid": true}`

2. **Instantiate Composite Pipeline**
   The LLM creates the full pipeline in a single call.
   ```json
   // instantiate
   {
     "estimator": ["ConditionalDeseasonalizer", "Detrender", "ARIMA"],
     "params": [{}, {}, {"order": [1, 1, 1]}]
   }
   ```
   *Returns:* `{"success": true, "estimator_handle": "est_pipe_789", "structure": "ConditionalDeseasonalizer ➔ Detrender ➔ ARIMA"}`

3. **Fit the Pipeline**
   ```json
   // fit
   {
     "estimator_handle": "est_pipe_789",
     "y": "airline"
   }
   ```

4. **Predict**
   ```json
   // predict
   {
     "estimator_handle": "est_pipe_789",
     "fh": 12
   }
   ```
   *Returns:* The predictions generated by the pipeline.


## 📁 Project Structure

```
sktime-mcp/
├── src/sktime_mcp/
│   ├── server.py           # MCP server entry point
│   ├── registry/           # Registry interface & tag resolver
│   ├── composition/        # Pipeline composition validator
│   ├── runtime/            # Execution engine, handle & job management
│   ├── data/               # Data adapters (file, pandas, SQL, URL)
│   └── tools/              # MCP tool implementations
├── docs/                   # Sphinx documentation source
├── examples/               # Usage examples
└── tests/                  # Test suite
```

## 🧪 Running Tests

```bash
pytest tests/
```

## Local Quality Checks

Run standardized local checks before raising a PR:

```bash
make check
```

Auto-fix formatting and fixable lint issues:

```bash
make format-fix
```

If `make` is unavailable (common on Windows), run the equivalent commands:

```bash
ruff format --check .
ruff check .
pytest
```

### Pre-Commit Hooks (Recommended)
To ensure your code meets quality standards before pushing, install the pre-commit hooks:
```bash
make install-hooks
```
This will automatically run Ruff and Pytest on your code every time you make a commit.