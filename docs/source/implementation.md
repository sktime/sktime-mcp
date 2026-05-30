# Complete Code Explanation: sktime-mcp

## 📋 Table of Contents
1. [Project Overview](#project-overview)
2. [Architecture](#architecture)
3. [File-by-File Breakdown](#file-by-file-breakdown)
4. [How It All Works Together](#how-it-all-works-together)
5. [Key Concepts](#key-concepts)

---

## Project Overview

**sktime-mcp** is a Model Context Protocol (MCP) server that exposes the `sktime` time series library to Large Language Models (LLMs). It allows LLMs to:

- **Discover** time series estimators from sktime's registry
- **Reason** about their capabilities using tags
- **Compose** estimators into pipelines
- **Execute** real forecasting workflows on datasets

### What Problem Does It Solve?

LLMs can't directly interact with Python libraries. This MCP server acts as a **semantic bridge**, translating between:
- **LLM world**: JSON-RPC requests with simple arguments
- **Python world**: Complex object instantiation, method calls, and data manipulation

---

## Architecture

The codebase is organized into **5 main layers**:

```
┌─────────────────────────────────────────┐
│         MCP Server (server.py)          │  ← Entry point, handles JSON-RPC
├─────────────────────────────────────────┤
│         Tools Layer (tools/)            │  ← MCP tool implementations
├─────────────────────────────────────────┤
│  Registry (registry/)  │  Composition   │  ← Discovery & Validation
│                        │  (composition/)│
├─────────────────────────────────────────┤
│         Runtime (runtime/)              │  ← Execution & Handle Management
├─────────────────────────────────────────┤
│            sktime Library               │  ← Actual ML library
└─────────────────────────────────────────┘
```

---

## File-by-File Breakdown

### 📁 Root Level Files

#### `README.md`
- **Purpose**: Project documentation and quick start guide
- **Key Sections**:
  - Installation instructions
  - Available MCP tools overview
  - Example LLM workflow
  - Project structure

#### `pyproject.toml`
- **Purpose**: Python project configuration (PEP 518)
- **Key Contents**:
  - Package metadata (name, version, description)
  - Dependencies: `mcp`, `sktime`, `pandas`, `numpy`, `scikit-learn`
  - Optional dependencies for dev and extended features
  - Entry point: `sktime-mcp` command → `sktime_mcp.server:main`
  - Tool configurations (ruff, pytest)

---

### 📁 `src/sktime_mcp/` - Core Source Code

#### `server.py` - MCP Server Entry Point
**Purpose**: Main MCP server that handles all tool calls

**Key Components**:
1. **`sanitize_for_json(obj)`**: Converts Python objects to JSON-serializable format
   - Handles numpy arrays, pandas objects, special types
   
2. **`@server.list_tools()`**: Registers all available MCP tools
   - Returns tool schemas (name, description, input schema)
   - Tools span Discovery, Instantiation, Execution, Data, Export, Persistence, Validation, and Job Management. (e.g., `list_estimators`, `instantiate_pipeline`, `fit_predict_async`, `load_data_source`, `save_model`, `check_job_status`).

3. **`@server.call_tool(name, arguments)`**: Routes tool calls to implementations
   - Validates arguments
   - Calls appropriate tool function
   - Sanitizes and returns results

4. **`main()`**: Entry point that starts the MCP server
   - Uses stdio transport (reads from stdin, writes to stdout)
   - Compatible with Claude Desktop and other MCP clients

**Flow**:
```
LLM → JSON-RPC request → server.call_tool() → tool function → sanitize → JSON response → LLM
```

---

### 📁 `src/sktime_mcp/registry/` - Estimator Discovery

#### `interface.py` - Registry Interface
**Purpose**: Wraps sktime's `all_estimators()` function and provides structured access

**Key Classes**:

1. **`EstimatorNode`** (dataclass)
   - Represents a single estimator with all its metadata
   - **Fields**:
     - `name`: Class name (e.g., "ARIMA")
     - `task`: Task type (e.g., "forecasting")
     - `module`: Python module path
     - `class_ref`: Actual Python class
     - `tags`: Capability tags (e.g., `{"capability:pred_int": True}`)
     - `hyperparameters`: Constructor parameters with defaults
     - `docstring`: Class documentation
   - **Methods**:
     - `to_dict()`: JSON serialization
     - `to_summary()`: Minimal info for list operations

2. **`RegistryInterface`** (singleton)
   - **Purpose**: Lazy-loads and caches all sktime estimators
   - **Key Methods**:
     - `get_all_estimators(task, tags)`: Filter estimators by task and tags
     - `get_estimator_by_name(name)`: Lookup specific estimator
     - `list_estimators(query=...)`: Text search in names/docstrings
     - `get_available_tasks()`: List all task types
     - `get_available_tags()`: List all capability tags
   - **Internal Methods**:
     - `_load_registry()`: Calls sktime's `all_estimators()` for each task
     - `_create_node()`: Extracts metadata from estimator class
     - `_get_tags()`: Calls `cls.get_class_tags()`
     - `_get_hyperparameters()`: Inspects `__init__` signature

**How It Works**:
```python
# First call triggers lazy loading
registry = get_registry()
registry._load_registry()  # Calls sktime.all_estimators("forecasting"), etc.

# Creates EstimatorNode for each estimator
for name, cls in estimators:
    node = EstimatorNode(
        name=name,
        task="forecasting",
        class_ref=cls,
        tags=cls.get_class_tags(),
        hyperparameters=inspect.signature(cls.__init__).parameters
    )
```

#### `tag_resolver.py` - Tag Resolution
**Purpose**: Handles tag-based filtering and compatibility checking

**Key Functions**:
- Resolves tag queries (e.g., `{"capability:pred_int": True}`)
- Checks if estimator tags match requirements
- Used by registry filtering and composition validation

---

### 📁 `src/sktime_mcp/composition/` - Pipeline Validation

#### `validator.py` - Composition Validator
**Purpose**: Validates that estimator compositions are valid before instantiation

**Key Classes**:

1. **`CompositionType`** (Enum)
   - Types of compositions: PIPELINE, TRANSFORMER_PIPELINE, FORECASTING_PIPELINE, MULTIPLEXER, ENSEMBLE, REDUCTION

2. **`CompositionRule`** (dataclass)
   - Defines valid composition patterns
   - Example: Transformers can precede forecasters

3. **`ValidationResult`** (dataclass)
   - **Fields**: `valid`, `errors`, `warnings`, `suggestions`
   - **Method**: `to_dict()` for JSON serialization

4. **`CompositionValidator`** (singleton)
   - **Key Methods**:
     - `validate_pipeline(components)`: Check if pipeline is valid
     - `_check_pair_compatibility(first, second)`: Validate two estimators can be composed
     - `_check_tag_compatibility(first, second)`: Check tag requirements
     - `get_valid_compositions(estimator_name)`: What can precede/follow this estimator
     - `suggest_pipeline(task, requirements)`: Suggest a valid pipeline

**Validation Rules**:
```python
# Valid: Transformer → Forecaster
["Detrender", "ARIMA"]  ✅

# Invalid: Forecaster → Forecaster
["ARIMA", "NaiveForecaster"]  ❌

# Valid: Multiple Transformers → Forecaster
["ConditionalDeseasonalizer", "Detrender", "ARIMA"]  ✅
```

---

### 📁 `src/sktime_mcp/runtime/` - Execution Engine

#### `handles.py` - Handle Manager
**Purpose**: Manages references to instantiated estimator objects

**Why Needed?**: 
- LLMs can't hold Python object references
- Solution: Create string handles (e.g., `"est_abc123"`) that map to objects

**Key Classes**:

1. **`HandleInfo`** (dataclass)
   - Stores metadata about a handle
   - **Fields**: `handle_id`, `estimator_name`, `instance`, `params`, `created_at`, `fitted`, `metadata`

2. **`HandleManager`** (singleton)
   - **Key Methods**:
     - `create_handle(estimator_name, instance, params)`: Create new handle → returns `"est_xyz"`
     - `get_instance(handle_id)`: Retrieve actual Python object
     - `get_info(handle_id)`: Get handle metadata
     - `mark_fitted(handle_id)`: Mark estimator as fitted
     - `is_fitted(handle_id)`: Check if fitted
     - `release_handle(handle_id)`: Free memory
     - `list_handles()`: List all active handles
     - `_cleanup_oldest()`: Auto-cleanup when max_handles reached

**Flow**:
```python
# Instantiation
instance = ARIMA(order=[1,1,1])
handle = manager.create_handle("ARIMA", instance, {"order": [1,1,1]})
# Returns: "est_a1b2c3d4e5f6"

# Later retrieval
instance = manager.get_instance("est_a1b2c3d4e5f6")
instance.fit(y)
```

#### `executor.py` - Execution Runtime
**Purpose**: Orchestrates estimator instantiation, data loading, fitting, and prediction

**Key Class**: `Executor` (singleton)

**Key Methods**:

1. **`instantiate(estimator_name, params)`**
   - Looks up estimator in registry
   - Instantiates with parameters
   - Creates handle
   - Returns: `{"success": True, "handle": "est_xyz", ...}`

2. **`load_dataset(name)`**
   - Loads demo datasets (airline, sunspots, etc.)
   - Uses sktime's dataset loaders
   - Returns: pandas Series/DataFrame

3. **`fit(handle_id, y, X, fh)`**
   - Retrieves instance from handle
   - Calls `instance.fit(y, X=X, fh=fh)`
   - Marks handle as fitted

4. **`predict(handle_id, fh, X)`**
   - Retrieves fitted instance
   - Calls `instance.predict(fh=fh, X=X)`
   - Returns predictions

5. **`fit_predict(handle_id, dataset, horizon)`**
   - Convenience method: load → fit → predict
   - Returns: `{"success": True, "predictions": {...}, "horizon": 12}`

6. **`instantiate_pipeline(components, params_list)`** ⭐ **Most Complex**
   - **Purpose**: Create complete pipelines from component names
   - **Steps**:
     1. Validate pipeline composition
     2. Instantiate each component
     3. Build `steps` argument: `[("name1", instance1), ("name2", instance2)]`
     4. Determine pipeline type (TransformedTargetForecaster, Pipeline, etc.)
     5. Instantiate pipeline with steps
     6. Create handle
   - **Why Complex**: Handles the "steps problem" - LLMs can't pass Python objects, so we build them server-side

**Example Flow**:
```python
# LLM sends:
{"components": ["Detrender", "ARIMA"], "params_list": [{}, {"order": [1,1,1]}]}

# Executor does:
detrender = Detrender()
arima = ARIMA(order=[1,1,1])
steps = [("transformer", detrender), ("forecaster", arima)]
pipeline = TransformedTargetForecaster(steps=steps)
handle = handle_manager.create_handle("Pipeline", pipeline)

# Returns to LLM:
{"success": True, "handle": "est_xyz", "pipeline": "Detrender → ARIMA"}
```

---

### 📁 `src/sktime_mcp/tools/` - MCP Tool Implementations

Each file implements one or more MCP tools that LLMs can call.

#### `list_estimators.py`
**Tools**:
1. **`list_estimators_tool(task, tags, query, limit)`**
   - Calls `registry.get_all_estimators(task, tags)`
   - Returns: `{"success": True, "estimators": [...], "total": 50}`

2. **`get_available_tags()`**
   - Returns all capability tags
   - Example: `["capability:pred_int", "handles-missing-data", ...]`

#### `describe_estimator.py`
**Tool**: `describe_estimator_tool(estimator)`
- Looks up estimator in registry
- Returns full EstimatorNode details
- Includes: name, task, module, tags, hyperparameters, docstring

#### `instantiate.py`
**Tools**:
1. **`instantiate_estimator_tool(estimator, params)`**
   - Calls `executor.instantiate(estimator, params)`
   - Returns handle

2. **`instantiate_pipeline_tool(components, params_list)`** ⭐
   - Calls `executor.instantiate_pipeline(components, params_list)`
   - Solves the "steps problem"
   - Returns single handle for entire pipeline

3. **`release_handle_tool(handle)`**
   - Frees memory for a handle

4. **`list_handles_tool()`**
   - Lists all active handles

5. **`load_model_tool(path)`**
   - Loads a previously saved model via MLflow

#### `fit_predict.py`
**Tools**:
1. **`fit_predict_tool(estimator_handle, dataset, horizon)`**
   - Calls `executor.fit_predict(handle, dataset, horizon)`
   - Complete workflow in one call

2. **`fit_predict_async_tool(estimator_handle, dataset, horizon)`**
   - Dispatches a background job for fit and predict.

#### `evaluate.py`
**Tool**: `evaluate_estimator_tool(estimator_handle, dataset, cv_folds)`
- Runs cross-validation using an expanding window splitter
- Returns comparison metrics like MAE and RMSE

#### `format_tools.py`
**Tools**:
1. **`format_time_series_tool(...)`**
   - Auto-formats, infers frequency, drops duplicates, and fills missing values.
2. **`auto_format_on_load_tool(enabled)`**
   - Toggles whether new data sources get auto-formatted on load.

#### `job_tools.py`
**Tools**: `check_job_status_tool`, `list_jobs_tool`, `cancel_job_tool`, `delete_job_tool`, `cleanup_old_jobs_tool`
- Interfaces with `JobManager` to control background training jobs.

#### `save_model.py`
**Tool**: `save_model_tool(estimator_handle, path, mlflow_params)`
- Persists fitted estimators using MLflow.

#### `list_available_data.py`
**Tool**: `list_available_data_tool(is_demo)`
   - Returns available demo datasets and/or active user data handles

---

### 📁 `examples/` - Usage Examples

#### `01_forecasting_workflow.py`
**Purpose**: Demonstrates all MCP capabilities end-to-end

**Steps**:
1. List datasets
2. Discover forecasting estimators
3. Filter by tags (probabilistic forecasters)
4. Describe an estimator
5. Validate pipeline compositions
6. Instantiate estimator
7. Fit and predict
8. List active handles
9. Show available tags

**Run**: `python examples/01_forecasting_workflow.py`

#### `02_llm_query_simulation.py`
**Purpose**: Simulates how an LLM would interact with the MCP

**Scenario**: User asks "Forecast airline passengers with a probabilistic model"

**LLM Steps**:
1. `list_estimators(task="forecasting", tags={"capability:pred_int": True})`
2. `describe_estimator("ARIMA")`
3. `instantiate_estimator("ARIMA", {"order": [1,1,1]})`
4. `fit_predict(handle, "airline", 12)`

#### `03_pipeline_instantiation.py`
**Purpose**: Demonstrates pipeline creation

**Examples**:
1. Simple 2-component pipeline
2. Complex 3-component pipeline (deseasonalize → detrend → forecast)
3. Pipeline with custom parameters
4. Invalid pipeline (shows validation errors)

#### `04_mcp_pipeline_demo.py`
**Purpose**: End-to-end pipeline workflow

**Steps**:
1. Validate pipeline
2. Instantiate pipeline → get handle
3. Fit and predict → get forecasts

#### Additional Examples
- `05_simple_deseasonalize_detrend_forecaster.py`: Deseasonalize + detrend workflow
- `06_simple_naive_forecaster.py`: Basic NaiveForecaster example
- `background_training_example.py`: Demonstrates async background jobs
- `job_management_demo.py`: Demonstrates checking and listing job status
- `pandas_example.py`: Demonstrates loading from in-memory pandas objects
- `csv_example.py`: Demonstrates loading from CSV/TSV files
- `sql_example.py`: Demonstrates loading from SQL databases

---

### 📁 `docs/` - Documentation

#### `architecture.md`
- **Purpose**: High-level block diagrams explaining the data flow and adapter registry.

#### `data-sources.md`
- **Purpose**: Detailed guide on loading data from Pandas, SQL, and various file formats.

#### `user-guide.md`
- **Purpose**: Information for end-users on how to use the MCP tools.

#### `dev-guide.md`
- **Purpose**: Guidelines for contributors on extending the server or adding new adapters.

---

### 📁 `tests/` - Test Suite

#### `test_core.py`
**Purpose**: Unit tests for core functionality

**Test Classes**:
1. **`TestRegistryInterface`**
   - Tests registry loading, filtering, lookup

2. **`TestHandleManager`**
   - Tests handle creation, retrieval, fitting, release

3. **`TestCompositionValidator`**
   - Tests pipeline validation logic

4. **`TestTools`**
   - Tests MCP tool functions

**Run**: `pytest tests/`

---

## How It All Works Together

### Example: LLM Forecasting Workflow

**User Prompt**: "Forecast airline passengers using ARIMA"

**Step 1: Discovery**
```
LLM → list_estimators(task="forecasting")
     → server.call_tool("list_estimators", {"task": "forecasting"})
     → list_estimators_tool(task="forecasting")
     → registry.get_all_estimators(task="forecasting")
     → Returns: [{"name": "ARIMA", ...}, {"name": "NaiveForecaster", ...}, ...]
```

**Step 2: Description**
```
LLM → describe_estimator("ARIMA")
     → describe_estimator_tool("ARIMA")
     → registry.get_estimator_by_name("ARIMA")
     → Returns: {"name": "ARIMA", "hyperparameters": {"order": ...}, ...}
```

**Step 3: Instantiation**
```
LLM → instantiate_estimator("ARIMA", {"order": [1,1,1]})
     → instantiate_estimator_tool("ARIMA", {"order": [1,1,1]})
     → executor.instantiate("ARIMA", {"order": [1,1,1]})
     → ARIMA_class = registry.get_estimator_by_name("ARIMA").class_ref
     → instance = ARIMA_class(order=[1,1,1])
     → handle = handle_manager.create_handle("ARIMA", instance)
     → Returns: {"success": True, "handle": "est_abc123"}
```

**Step 4: Execution**
```
LLM → fit_predict("est_abc123", "airline", 12)
     → fit_predict_tool("est_abc123", "airline", 12)
     → executor.fit_predict("est_abc123", "airline", 12)
     → y = executor.load_dataset("airline")
     → instance = handle_manager.get_instance("est_abc123")
     → instance.fit(y)
     → predictions = instance.predict(fh=[1,2,...,12])
     → Returns: {"success": True, "predictions": {...}, "horizon": 12}
```

### Data Flow Diagram

```
┌─────────┐
│   LLM   │
└────┬────┘
     │ JSON-RPC request
     ▼
┌─────────────────┐
│  MCP Server     │ ← server.py
│  (stdio)        │
└────┬────────────┘
     │ Route to tool
     ▼
┌─────────────────┐
│  Tool Function  │ ← tools/*.py
└────┬────────────┘
     │ Call business logic
     ▼
┌──────────────────────────────────┐
│  Registry / Executor / Validator │ ← registry/, runtime/, composition/
└────┬─────────────────────────────┘
     │ Interact with sktime
     ▼
┌─────────────────┐
│  sktime Library │
└─────────────────┘
```

---

## Key Concepts

### 1. **Registry-First Design**
- Don't parse code or docs
- Use sktime's `all_estimators()` as source of truth
- Extract metadata from classes directly

### 2. **Handle-Based References**
- LLMs can't hold Python objects
- Solution: String handles (`"est_abc123"`) map to objects
- Handle manager maintains the mapping

### 3. **Lazy Loading**
- Registry loads on first access
- Singleton pattern ensures one instance
- Caches all estimators for fast lookups

### 4. **Tag-Based Discovery**
- Estimators have capability tags
- LLMs can filter by requirements
- Example: `{"capability:pred_int": True}` finds probabilistic forecasters

### 5. **Composition Validation**
- Check pipeline validity before instantiation
- Prevents runtime errors
- Provides helpful error messages

### 6. **The Steps Problem**
- **Problem**: Pipelines need `steps=[("name", instance), ...]`
- **Solution**: LLM sends component names, server builds instances
- **Benefit**: LLM uses simple JSON, server handles complexity

### 7. **JSON Sanitization**
- Convert numpy/pandas to JSON-serializable types
- Handle special values (NaN, Infinity)
- Ensure all responses are valid JSON

### 8. **Singleton Pattern**
- Registry, Executor, HandleManager, Validator are singletons
- Ensures shared state across tool calls
- Efficient memory usage

---

## Summary

**sktime-mcp** is a well-architected MCP server that:

1. **Exposes** sktime's 200+ estimators to LLMs
2. **Validates** compositions before execution
3. **Manages** object lifecycles via handles
4. **Executes** real ML workflows on real data
5. **Translates** between JSON (LLM) and Python (sktime)

**Key Innovation**: The `instantiate_pipeline` tool solves the "steps problem", enabling LLMs to create complex pipelines with a single JSON-RPC call.

**Architecture Highlights**:
- Clean separation of concerns (registry, composition, runtime, tools)
- Singleton pattern for shared state
- Handle-based object management
- Comprehensive validation before execution
- JSON-first API design

This enables LLMs to perform sophisticated time series forecasting workflows without writing any Python code! 🚀
