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

```bash
# Install from source
pip install -e .

# With all optional dependencies
pip install -e ".[all]"

# Development installation
pip install -e ".[dev]"
```

## 🚀 Quick Start

### Running the MCP Server

```bash
# Start the MCP server
sktime-mcp

# Or run directly
python -m sktime_mcp.server
```

### Connecting from an LLM Client

The server uses stdio transport by default, compatible with Claude Desktop and other MCP clients.

Add to your Claude Desktop config (`~/.config/claude/claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "sktime": {
      "command": "sktime-mcp"
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

#### 2. `search_estimators`
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
Execute a complete workflow: load dataset, fit estimator, and generate predictions.

**Arguments:**
- `estimator_handle` (required): Handle from `instantiate_estimator` or `instantiate_pipeline`
- `dataset` (required): Dataset name (e.g., `"airline"`, `"sunspots"`, `"lynx"`)
- `horizon` (optional): Forecast horizon (default: 12)

**Example:**
```json
{
  "estimator_handle": "est_abc123",
  "dataset": "airline",
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

### Datasets

#### 10. `list_datasets`
List all available demo datasets for testing and experimentation.

**Arguments:** None

**Returns:** `{"success": true, "datasets": ["airline", "sunspots", "lynx", "shampoo", ...]}`

---

### Handle Management

#### 11. `list_handles`
List all active estimator handles and their status.

**Arguments:** None

**Returns:** List of active handles with metadata (estimator name, fitted status, creation time)

---

#### 12. `release_handle`
Release an estimator handle and free memory.

**Arguments:**
- `handle` (required): Handle ID to release

**Example:**
```json
{
  "handle": "est_abc123"
}
```

**Returns:** `{"success": true, "message": "Handle released"}`

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
sktime_mcp/
├── src/sktime_mcp/
│   ├── server.py           # MCP server entry point
│   ├── registry/           # Registry interface & tag resolver
│   ├── composition/        # Pipeline composition validator
│   ├── runtime/            # Execution engine & handle management
│   └── tools/              # MCP tool implementations
├── examples/               # Usage examples
└── tests/                  # Test suite
```

## 🧪 Running Tests

```bash
pytest tests/
```
