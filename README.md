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

## 🛠️ Prerequisites

- **Python 3.10+**
- **pip** package manager

## 🛠️ Installation

### Virtual Environment Setup

It is recommended to use a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### Package Installation

The recommended way to install is using `python3 -m pip`:

```bash
# Install from source
python3 -m pip install -e .

# With all optional dependencies
python3 -m pip install -e ".[all]"

# Development installation
python3 -m pip install -e ".[dev]"
```

Alternatively, you can use `pip`:

```bash
# Install from source
pip install -e .

# With all optional dependencies
pip install -e ".[all]"

# Development installation
pip install -e ".[dev]"
```

For a more detailed first-time setup flow, including MCP server verification and troubleshooting, see [Beginner Setup](#-beginner-setup-firsttime-users).

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
| `ModuleNotFoundError: sktime_mcp` | Package not installed in the active environment | Confirm `.venv` is active (`which python`) and re-run `pip install -e ".[dev]"` |
| `pip: command not found` | System `pip` not available | Use `python -m pip` instead of bare `pip` |
| Wrong Python version selected | Multiple Python installations | Invoke `python3 -m venv .venv` explicitly and always use `python` inside the activated environment |

### Minimal Setup Check

After completing the steps above, confirm the server starts with `sktime-mcp`. See the [macOS / Unix-like first-time setup](#macos--unix-like-first-time-setup) section for the fallback command and common error solutions.

> **Note:** On Windows, the `sktime-mcp` command may be installed to a directory
> not on your `PATH` (e.g., `%APPDATA%\Python\Python3xx\Scripts`). Either add
> that directory to your `PATH` or use `python -m sktime_mcp.server` instead.


## 🚀 Quick Start

### Running the MCP Server

```bash
sktime-mcp
```

If `sktime-mcp` is not on your `PATH`, use `python -m sktime_mcp.server` instead (see [Beginner Setup](#-beginner-setup-firsttime-users) for details).

### Connecting from an LLM Client

The server uses stdio transport by default, compatible with Claude Desktop, Claude Code, and other MCP clients.

**Claude Desktop** — add to your config file:

| Platform | Config path |
|----------|-------------|
| macOS    | `~/Library/Application Support/Claude/claude_desktop_config.json` |
| Linux    | `~/.config/claude/claude_desktop_config.json` |
| Windows  | `%APPDATA%\Claude\claude_desktop_config.json` |

```json
{
  "mcpServers": {
    "sktime": {
      "command": "sktime-mcp"
    }
  }
}
```

If you are using a virtual environment, or if `sktime-mcp` is not on your `PATH`, point the client directly at the environment's Python executable — this ensures the correct packages are used:

```json
{
  "mcpServers": {
    "sktime": {
      "command": "<project-root>/.venv/bin/python",
      "args": ["-m", "sktime_mcp.server"]
    }
  }
}
```

## 📚 Available Tools

### Discovery & Search

#### 1. `list_estimators`
Discover estimators by task type and capability tags.

**Arguments:**
- `task` (optional): Task type filter (`"forecasting"`, `"classification"`, `"regression"`, `"transformation"`, `"clustering"`)
- `tags` (optional): Filter by capability tags (e.g., `{"capability:pred_int": true}`)
- `limit` (optional): Maximum results (default: 50)

**Example:**
```json
{
  "task": "forecasting",
  "tags": {
    "capability:pred_int": true
  },
  "limit": 10
}
```

**Returns:** List of matching estimators with name, task, and summary info.

---

#### 2. `list_estimators` (query mode)
Search estimators by name or description using text query.

**Arguments:**
- `query` (required): Search string (case-insensitive)
- `limit` (optional): Maximum results (default: 20)

**Example:**
```json
{
  "query": "ARIMA",
  "limit": 5
}
```

**Returns:** List of estimators matching the search query.

---

#### 3. `describe_estimator`
Get detailed information about a specific estimator's capabilities.

**Arguments:**
- `estimator` (required): Name of the estimator (e.g., `"ARIMA"`, `"NaiveForecaster"`)

**Example:**
```json
{
  "estimator": "ARIMA"
}
```

**Returns:** Full estimator details including tags, hyperparameters, docstring, and module path.

---

#### 4. `get_available_tags`
List all queryable capability tags across all estimators.

**Arguments:** None

**Returns:** List of all available tags (e.g., `["capability:pred_int", "handles-missing-data", ...]`)

---

### Instantiation

#### 5. `instantiate_estimator`
Create a single estimator instance and return a handle.

**Arguments:**
- `estimator` (required): Name of the estimator to instantiate
- `params` (optional): Hyperparameters for the estimator

**Example:**
```json
{
  "estimator": "ARIMA",
  "params": {
    "order": [1, 1, 1],
    "suppress_warnings": true
  }
}
```

**Returns:** `{"success": true, "handle": "est_abc123", "estimator": "ARIMA", "params": {...}}`

---

#### 6. `instantiate_pipeline` 
Create a complete pipeline from a list of components (transformers → forecaster).

**Arguments:**
- `components` (required): List of estimator names in pipeline order
- `params_list` (optional): List of parameter dicts for each component

**Example:**
```json
{
  "components": ["ConditionalDeseasonalizer", "Detrender", "ARIMA"],
  "params_list": [{}, {}, {"order": [1, 1, 1]}]
}
```

**Returns:** `{"success": true, "handle": "est_xyz789", "pipeline": "ConditionalDeseasonalizer → Detrender → ARIMA", ...}`

**Note:** This solves the "steps problem" - you don't need to instantiate components separately!

---

## 📖 Documentation

Project documentation lives in `docs/` and can be served locally with MkDocs:

```bash
pip install -e ".[dev]"
mkdocs serve
```

The MkDocs config is in `mkdocs.yml`.

### Validation

#### 7. `validate_pipeline`
Check if a proposed pipeline composition is valid before instantiation.

**Arguments:**
- `components` (required): List of estimator names in pipeline order

**Example:**
```json
{
  "components": ["Detrender", "ARIMA"]
}
```

**Returns:** `{"valid": true/false, "errors": [...], "warnings": [...], "suggestions": [...]}`

---

### Execution

#### 8. `fit_predict`
Execute a complete workflow: fit the estimator and generate predictions. Use a **demo dataset name** or a **`data_handle`** from `load_data_source` (provide exactly one of the two).

**Arguments:**
- `estimator_handle` (required): Handle from `instantiate_estimator` or `instantiate_pipeline`
- `dataset` (optional): Demo dataset name (e.g., `"airline"`, `"sunspots"`, `"lynx"`) when not using custom data
- `data_handle` (optional): Handle from `load_data_source` for custom data (omit `dataset` in that case)
- `horizon` (optional): Forecast horizon (default: 12)

**Example (demo data):**
```json
{
  "estimator_handle": "est_abc123",
  "dataset": "airline",
  "horizon": 12
}
```

**Example (custom data):**
```json
{
  "estimator_handle": "est_abc123",
  "data_handle": "data_abc123",
  "horizon": 12
}
```

**Returns:** `{"success": true, "predictions": {1: 450.2, 2: 455.1, ...}, "horizon": 12}`

---

#### 9. `save_model`
Persist a fitted estimator or pipeline handle to a local filesystem path using `sktime.utils.mlflow_sktime.save_model`.

**Arguments:**
- `estimator_handle` (required): Handle from `instantiate_estimator` or `instantiate_pipeline`
- `path` (required): Local filesystem path where the model should be saved
- `mlflow_params` (optional): Extra keyword arguments forwarded to `sktime.utils.mlflow_sktime.save_model`

**Example:**
```json
{
  "estimator_handle": "est_abc123",
  "path": "/absolute/path/to/model_dir",
  "mlflow_params": {
    "serialization_format": "cloudpickle"
  }
}
```

**Returns:** `{"success": true, "saved_path": "/absolute/path/to/model_dir", "message": "Model saved successfully to '/absolute/path/to/model_dir'"}`

**Note:** This tool requires MLflow to be available in the server environment.

---

### Data Availability

#### 10. `list_available_data`
List all available data — both system demo datasets and active user-loaded data handles — in a single unified response.

**Arguments:**
- `is_demo` (optional): `true` returns only demo datasets, `false` returns only active data handles, omit to get both

**Returns:** `{"success": true, "system_demos": ["airline", "sunspots", ...], "active_handles": [...], "total": 8}`

---

### Data Loading

#### 11. `load_data_source`
Load data from various sources (CSV/Parquet files, pandas DataFrames, SQL databases, URLs).

**Arguments:**
- `config` (required): Data source configuration object. Must include `"type"` (`"pandas"`, `"sql"`, `"file"`, `"url"`).

**Example:**
```json
{
  "config": {
    "type": "file",
    "path": "/absolute/path/to/data.csv",
    "time_column": "date",
    "target_column": "value"
  }
}
```

**Returns:** `{"success": true, "data_handle": "data_abc123", "metadata": {...}}`

---

### Code & Model Export

#### 12. `export_code`
Export an estimator or pipeline as executable Python code.

**Arguments:**
- `handle` (required): Handle ID of the estimator/pipeline
- `var_name` (optional): Variable name in generated code (default: `"model"`)
- `include_fit_example` (optional): Include a fit/predict example (default: `false`)

**Returns:** `{"success": true, "code": "from sktime..."}`

---

### Background Jobs

#### 13. `fit_predict_async`
Non-blocking version of `fit_predict`. Returns a `job_id` immediately; use `check_job_status` to poll.

**Arguments:** Same as `fit_predict`.

**Returns:** `{"success": true, "job_id": "abc-123", "message": "Training job started..."}`

---

#### 14. `check_job_status`
Check the status and progress of a background job.

**Arguments:**
- `job_id` (required): Job ID to check

---

#### 15. `list_jobs`
List all background jobs with optional status filter.

**Arguments:**
- `status` (optional): Filter by status (`"pending"`, `"running"`, `"completed"`, `"failed"`, `"cancelled"`)
- `limit` (optional): Maximum results (default: 20)

---

### Data Formatting

#### 16. `format_time_series`
Automatically format time series data (infer frequency, remove duplicates, fill missing values).

**Arguments:**
- `data_handle` (required): Handle from `load_data_source`
- `auto_infer_freq` (optional): Infer and set frequency (default: `true`)
- `fill_missing` (optional): Fill missing values (default: `true`)
- `remove_duplicates` (optional): Remove duplicate timestamps (default: `true`)

## 🔄 Example LLM Flows

### Flow 1: Simple Forecasting

**User Prompt:** "Forecast monthly airline passengers using a probabilistic model."

**LLM Steps:**

1. **Discover Models**
   ```
   list_estimators(task="forecasting", tags={"capability:pred_int": true})
   ```

2. **Inspect Choice**
   ```
   describe_estimator(estimator="ARIMA")
   ```

3. **Instantiate**
   ```
   instantiate_estimator(estimator="ARIMA", params={"order": [1,1,1]})
   → Returns: {"handle": "est_abc123"}
   ```

4. **Execute**
   ```
   fit_predict(estimator_handle="est_abc123", dataset="airline", horizon=12)
   → Returns: {"predictions": {1: 450.2, 2: 455.1, ...}}
   ```

### Flow 2: Pipeline Forecasting ⭐

**User Prompt:** "Forecast with deseasonalization and detrending preprocessing."

**LLM Steps:**

1. **Validate Composition**
   ```
   validate_pipeline(components=["ConditionalDeseasonalizer", "Detrender", "ARIMA"])
   → Returns: {"valid": true}
   ```

2. **Instantiate Pipeline** (single call!)
   ```
   instantiate_pipeline(
     components=["ConditionalDeseasonalizer", "Detrender", "ARIMA"],
     params_list=[{}, {}, {"order": [1,1,1]}]
   )
   → Returns: {"handle": "est_xyz789", "pipeline": "ConditionalDeseasonalizer → Detrender → ARIMA"}
   ```

3. **Execute**
   ```
   fit_predict(estimator_handle="est_xyz789", dataset="airline", horizon=12)
   → Returns: {"predictions": {...}}
   ```

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
├── docs/                   # MkDocs documentation source
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
python -m black --check .
python -m ruff check .
python -m pytest
```
