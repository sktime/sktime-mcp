# sktime-mcp: Comprehensive System Architecture & Review

> **Revision date:** April 2026
> **Scope:** Full codebase review covering architecture, tool inventory, testing strategy, documentation quality, performance, and improvement roadmap.

---

## Table of Contents

1. [High-Level Architecture](#1-high-level-architecture)
2. [Repository Layout](#2-repository-layout)
3. [Layer-by-Layer Deep Dive](#3-layer-by-layer-deep-dive)
   - 3.1 [MCP Server (Entry Point)](#31-mcp-server-entry-point)
   - 3.2 [Tool Layer](#32-tool-layer)
   - 3.3 [Registry Layer](#33-registry-layer)
   - 3.4 [Runtime Layer](#34-runtime-layer)
   - 3.5 [Data Layer](#35-data-layer)
   - 3.6 [Composition Layer](#36-composition-layer)
4. [Complete Tool Inventory](#4-complete-tool-inventory)
5. [Data Flow Diagrams](#5-data-flow-diagrams)
6. [Tool Combination Strategies](#6-tool-combination-strategies)
7. [Current Test Coverage & Testing Strategy](#7-current-test-coverage--testing-strategy)
8. [Documentation Review](#8-documentation-review)
9. [Bugs & Issues Found](#9-bugs--issues-found)
10. [Performance Analysis & Improvements](#10-performance-analysis--improvements)
11. [Improvement Roadmap](#11-improvement-roadmap)
12. [Appendix: File-by-File Reference](#appendix-file-by-file-reference)

---

## 1. High-Level Architecture

sktime-mcp is a **Model Context Protocol (MCP) server** that wraps the [sktime](https://www.sktime.net/) time-series machine learning library, exposing its estimator registry and execution capabilities to LLM agents (e.g., Claude, ChatGPT) via a structured JSON tool interface.

```
┌───────────────────────────────────────────────────────────────────┐
│                        LLM / MCP Client                          │
│              (Claude Desktop, Cursor, custom agent)               │
└───────────────────────┬───────────────────────────────────────────┘
                        │  stdio (JSON-RPC over stdin/stdout)
                        ▼
┌───────────────────────────────────────────────────────────────────┐
│                      MCP Server (server.py)                       │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────────────┐   │
│  │ list_tools() │  │ call_tool()  │  │ sanitize_for_json()    │   │
│  │ (28 tools)   │  │ (dispatcher) │  │ (serialization helper) │   │
│  └─────────────┘  └──────┬───────┘  └────────────────────────┘   │
└──────────────────────────┼────────────────────────────────────────┘
                           │
          ┌────────────────┼────────────────────┐
          ▼                ▼                     ▼
┌──────────────┐  ┌───────────────┐   ┌──────────────────┐
│  Tool Layer  │  │ Registry Layer│   │  Composition     │
│  (tools/*.py)│  │ (registry/)   │   │  Layer           │
│              │  │               │   │  (composition/)  │
│ 12 modules   │  │ interface.py  │   │  validator.py    │
│ 28+ functions│  │ tag_resolver  │   │                  │
└──────┬───────┘  └───────┬───────┘   └──────────────────┘
       │                  │
       ▼                  ▼
┌──────────────────────────────────────────────────────────────────┐
│                       Runtime Layer                               │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────────────────┐  │
│  │  Executor     │  │ HandleManager│  │   JobManager           │  │
│  │  (executor.py)│  │ (handles.py) │  │   (jobs.py)            │  │
│  │              │  │              │  │                        │  │
│  │ fit/predict   │  │ est_* handles│  │ background jobs        │  │
│  │ data loading  │  │ max 100      │  │ progress tracking      │  │
│  │ pipelines     │  │ LRU eviction │  │ thread-safe            │  │
│  └──────┬───────┘  └──────────────┘  └────────────────────────┘  │
└─────────┼────────────────────────────────────────────────────────┘
          │
          ▼
┌──────────────────────────────────────────────────────────────────┐
│                         Data Layer                                │
│  ┌────────────┐  ┌────────────────────────────────────────────┐  │
│  │  Registry   │  │              Adapters                      │  │
│  │  (registry) │  │  ┌─────────┐ ┌──────┐ ┌─────┐ ┌───────┐  │  │
│  │             │  │  │ Pandas  │ │ File │ │ SQL │ │  URL  │  │  │
│  └────────────┘  │  └─────────┘ └──────┘ └─────┘ └───────┘  │  │
│                  └────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────┘
          │
          ▼
┌──────────────────────────────────────────────────────────────────┐
│                     sktime Library                                │
│  all_estimators() · all_tags() · datasets · forecasting · etc.   │
└──────────────────────────────────────────────────────────────────┘
```

### Design Principles

| Principle | Implementation |
|-----------|---------------|
| **sktime as single source of truth** | Registry loads all estimators from `sktime.registry.all_estimators()` at runtime; no hardcoded estimator lists. |
| **Registry-first discovery** | LLMs discover estimators via `list_estimators` / `search_estimators` / `describe_estimator` before using them. |
| **Handle-based state** | Estimators and data are referenced by opaque `est_*` / `data_*` handle strings, not serialized objects. |
| **Lazy loading** | Registry, tag resolver, and executor are singletons initialized on first access. |
| **Fail-safe responses** | Every tool returns `{"success": bool, ...}` — never bare exceptions over the wire. |

---

## 2. Repository Layout

```
sktime-mcp/
├── pyproject.toml              # Build config, deps, extras, lint/test settings
├── Makefile                    # Developer shortcuts (check, lint, format, test)
├── mkdocs.yml                  # Documentation site configuration
├── README.md                   # Primary documentation
├── LICENSE                     # BSD 3-Clause
├── .gitignore
│
├── .github/
│   └── workflows/
│       ├── ci.yml              # CI: lint + test (3.9-3.12) + docs build
│       └── PULL_REQUEST_TEMPLATE.md  # (misplaced — see §9)
│
├── src/sktime_mcp/             # Installable Python package
│   ├── __init__.py
│   ├── server.py               # MCP server entry point (tool registration + dispatch)
│   ├── tools/                  # Tool implementations (12 modules)
│   ├── registry/               # sktime registry bridge (interface + tags)
│   ├── runtime/                # Executor, handles, jobs
│   ├── composition/            # Pipeline validator
│   └── data/                   # Data adapters (pandas, file, SQL, URL)
│
├── tests/                      # pytest suite (7 test modules)
├── examples/                   # 11 runnable example scripts
├── docs/                       # MkDocs source pages (9 markdown files)
└── site/                       # Built static site (should be in .gitignore)
```

---

## 3. Layer-by-Layer Deep Dive

### 3.1 MCP Server (Entry Point)

**File:** `src/sktime_mcp/server.py` (742 lines)

The server is the single entry point. It uses the official `mcp` Python SDK and communicates over stdio (JSON-RPC).

**Key Components:**
- `server = Server("sktime-mcp")` — MCP server instance
- `@server.list_tools()` — Returns a list of 28 `Tool` objects with hand-written JSON schemas
- `@server.call_tool()` — A large `if/elif` chain dispatching tool names to Python functions
- `sanitize_for_json()` — Recursive serializer for non-JSON-safe objects (Period, numpy types, etc.)
- `main()` → `asyncio.run(run_server())` — CLI entry point

**Architecture Concern:** The server file combines schema definition, dispatch logic, and serialization in one 742-line file. As the tool count grows, this becomes a maintenance bottleneck.

### 3.2 Tool Layer

**Location:** `src/sktime_mcp/tools/` (12 modules)

Each module contains one or more tool functions. Tools are thin wrappers that validate input, delegate to the runtime layer, and format output.

| Module | Tools Defined | MCP-Registered |
|--------|--------------|----------------|
| `list_estimators.py` | `list_estimators_tool`, `get_available_tags`, `get_available_tasks` | `list_estimators`, `get_available_tags` |
| `describe_estimator.py` | `describe_estimator_tool`, `search_estimators_tool` | `describe_estimator`, `search_estimators` |
| `instantiate.py` | `instantiate_estimator_tool`, `instantiate_pipeline_tool`, `list_handles_tool`, `release_handle_tool`, `load_model_tool` | All 5 |
| `fit_predict.py` | `fit_predict_tool`, `fit_predict_async_tool`, `fit_tool`, `predict_tool`, `list_datasets_tool` | `fit_predict`, `fit_predict_async` |
| `evaluate.py` | `evaluate_estimator_tool` | `evaluate_estimator` |
| `data_tools.py` | `load_data_source_tool`, `list_data_sources_tool`, `fit_predict_with_data_tool`, `release_data_handle_tool`, `load_data_source_async_tool` | All 5 |
| `format_tools.py` | `format_time_series_tool`, `auto_format_on_load_tool` | Both |
| `codegen.py` | `export_code_tool`, `_generate_single_estimator_code`, `_generate_pipeline_code` | `export_code` |
| `job_tools.py` | `check_job_status_tool`, `list_jobs_tool`, `cancel_job_tool`, `delete_job_tool`, `cleanup_old_jobs_tool` | All 5 |
| `save_model.py` | `save_model_tool` | `save_model` |
| `list_available_data.py` | `list_available_data_tool` | `list_available_data` |

**Not MCP-registered:** `fit_tool`, `predict_tool`, `list_datasets_tool`, `get_available_tasks` — these are library-only helpers.

### 3.3 Registry Layer

**Location:** `src/sktime_mcp/registry/`

| File | Purpose |
|------|---------|
| `interface.py` | `RegistryInterface` — wraps `sktime.registry.all_estimators()`. Lazy-loads all estimator types, creates `EstimatorNode` dataclasses with name, task, class reference, tags, hyperparameters, and docstring. Supports filtering by task, tags, and substring search. |
| `tag_resolver.py` | `TagResolver` — loads `sktime.registry.all_tags()`, provides human-readable tag explanations, category grouping, capability filtering, similarity scoring. |

**Singleton pattern:** `get_registry()` / `get_tag_resolver()` — process-wide.

**`EstimatorNode` dataclass fields:**
- `name` (str) — class name, e.g., "ARIMA"
- `task` (str) — mapped from scitype: "forecasting", "transformation", "classification", "regression", "clustering", "parameter_estimation", "splitting", "network"
- `class_ref` (type) — actual Python class
- `module` (str) — full dotted path
- `tags` (dict) — capability tags from `get_class_tags()`
- `hyperparameters` (dict) — extracted from `__init__` signature
- `docstring` (str) — truncated to 500 chars

### 3.4 Runtime Layer

**Location:** `src/sktime_mcp/runtime/`

#### Executor (`executor.py`)

The central runtime component that orchestrates all execution:

- **Estimator instantiation** — `instantiate()`, `instantiate_pipeline()`
- **Demo dataset loading** — Dynamic discovery via `inspect.getmembers(sktime.datasets)` → `DEMO_DATASETS` dict
- **Fit/predict** — `fit()`, `predict()`, `fit_predict()`, `fit_predict_with_data()`
- **Async execution** — `fit_predict_async()`, `load_data_source_async()` using `asyncio.run_in_executor()`
- **Data source management** — `load_data_source()`, `format_data_handle()`, `list_data_handles()`, `release_data_handle()`

Singleton: `get_executor()`

#### Handle Manager (`handles.py`)

In-memory store for estimator instances:

- **Handle format:** `est_{12-char-hex}`
- **Max capacity:** 100 handles; LRU eviction (drops oldest 10 when full)
- **Tracks:** instance reference, creation time, fitted status, metadata
- **No persistence** — handles are lost on server restart

#### Job Manager (`jobs.py`)

Thread-safe background job tracker:

- **Job states:** PENDING → RUNNING → COMPLETED / FAILED / CANCELLED
- **Progress tracking:** step count, percentage, ETA estimation
- **Thread safety:** all operations protected by `threading.Lock`
- **Limitation:** `cancel_job` only flips status flag; does not interrupt running asyncio tasks

### 3.5 Data Layer

**Location:** `src/sktime_mcp/data/`

**Abstract base:** `DataSourceAdapter` (in `base.py`) defines the interface:
- `load()` → `pd.DataFrame`
- `validate()` → `(bool, dict)`
- `to_sktime_format()` → `(pd.Series, Optional[pd.DataFrame])`
- `get_metadata()` → `dict`

**Adapters:**

| Adapter | Source | Key Features |
|---------|--------|--------------|
| `PandasAdapter` | In-memory dict/DataFrame | Auto datetime parsing, frequency inference |
| `FileAdapter` | CSV, Excel, Parquet files | Format auto-detection from extension, parse_dates, csv_options |
| `SQLAdapter` | SQL databases via SQLAlchemy | Query execution, identifier sanitization, connection string sanitization in metadata |
| `UrlAdapter` | HTTP/HTTPS URLs | Downloads to temp dir, delegates to FileAdapter, auto-cleanup |

**Data handle format:** `data_{8-char-hex}`

**No cap on data handles** (unlike estimator handles which cap at 100).

### 3.6 Composition Layer

**Location:** `src/sktime_mcp/composition/`

**`CompositionValidator`** validates pipeline compositions at planning time:

- **Composition rules:** Pairwise task compatibility (transformer→forecaster, transformer→transformer, classifier ensembles, etc.)
- **Tag compatibility:** Checks univariate vs. multivariate mismatches
- **Pipeline types supported:**
  - `TransformedTargetForecaster` (transformers → forecaster)
  - `Pipeline` (transformers → classifier/regressor)
  - `TransformerPipeline` (all transformers)
- **Suggestions:** Provides actionable fix suggestions for invalid compositions

---

## 4. Complete Tool Inventory

All 28 MCP-registered tools, categorized:

### Discovery Tools (4)
| Tool | Description | Parameters |
|------|-------------|------------|
| `list_estimators` | List estimators by task/tags | `task?`, `tags?`, `limit?` (default 50) |
| `describe_estimator` | Full metadata for one estimator | `estimator` (required) |
| `search_estimators` | Substring search on name/docstring | `query` (required), `limit?` (default 20) |
| `get_available_tags` | Tag catalog with descriptions | (none) |

### Instantiation Tools (5)
| Tool | Description | Parameters |
|------|-------------|------------|
| `instantiate_estimator` | Create estimator instance → handle | `estimator` (required), `params?` |
| `instantiate_pipeline` | Create pipeline instance → handle | `components` (required), `params_list?` |
| `list_handles` | List active estimator handles | (none) |
| `release_handle` | Delete estimator handle | `handle` (required) |
| `load_model` | Load saved model → handle | `path` (required) |

### Execution Tools (4)
| Tool | Description | Parameters |
|------|-------------|------------|
| `fit_predict` | Fit + predict on demo or custom data | `estimator_handle` (required), `dataset?`, `data_handle?`, `horizon?` |
| `fit_predict_async` | Background fit + predict | `estimator_handle` (required), `dataset` (required), `horizon?` |
| `fit_predict_with_data` | Fit + predict on custom data handle | `estimator_handle` (required), `data_handle` (required), `horizon?` |
| `evaluate_estimator` | Cross-validation evaluation | `estimator_handle` (required), `dataset` (required), `cv_folds?` |

### Data Tools (7)
| Tool | Description | Parameters |
|------|-------------|------------|
| `list_available_data` | List demos + active data handles | `is_demo?` |
| `load_data_source` | Load data from any adapter | `config` (required) |
| `load_data_source_async` | Background data loading | `config` (required) |
| `list_data_sources` | List adapter types | (none) |
| `release_data_handle` | Delete data handle | `data_handle` (required) |
| `format_time_series` | Auto-format time series data | `data_handle` (required), `auto_infer_freq?`, `fill_missing?`, `remove_duplicates?` |
| `auto_format_on_load` | Toggle auto-formatting | `enabled?` |

### Export Tools (1)
| Tool | Description | Parameters |
|------|-------------|------------|
| `export_code` | Generate Python code for handle | `handle` (required), `var_name?`, `include_fit_example?`, `dataset?` |

### Persistence Tools (2)
| Tool | Description | Parameters |
|------|-------------|------------|
| `save_model` | Save model via MLflow | `estimator_handle` (required), `path` (required), `mlflow_params?` |
| `load_model` | Load saved model | `path` (required) |

### Validation Tools (1)
| Tool | Description | Parameters |
|------|-------------|------------|
| `validate_pipeline` | Check pipeline validity | `components` (required) |

### Job Management Tools (4)
| Tool | Description | Parameters |
|------|-------------|------------|
| `check_job_status` | Get job progress/result | `job_id` (required) |
| `list_jobs` | List jobs with optional filter | `status?`, `limit?` |
| `cancel_job` | Cancel pending/running job | `job_id` (required) |
| `delete_job` | Remove job record | `job_id` (required) |
| `cleanup_old_jobs` | Remove old jobs | `max_age_hours?` (default 24) |

---

## 5. Data Flow Diagrams

### Workflow 1: Discovery → Instantiation → Prediction

```
LLM                        MCP Server                      sktime
 │                              │                              │
 │ list_estimators(task="forecasting")                         │
 │─────────────────────────────>│                              │
 │                              │ RegistryInterface            │
 │                              │ .get_all_estimators()        │
 │                              │ ────────────────────────────>│
 │                              │<─────(EstimatorNode list)────│
 │<───── {estimators: [...]} ───│                              │
 │                              │                              │
 │ describe_estimator("ARIMA")  │                              │
 │─────────────────────────────>│                              │
 │<── {hyperparameters, tags} ──│                              │
 │                              │                              │
 │ instantiate_estimator(       │                              │
 │   "ARIMA", {"order":[1,1,1]})│                             │
 │─────────────────────────────>│                              │
 │                              │ Executor.instantiate()       │
 │                              │  → ARIMA(order=[1,1,1])      │
 │                              │  → HandleManager.create()    │
 │<──── {handle: "est_abc123"} ─│                              │
 │                              │                              │
 │ fit_predict(                 │                              │
 │   handle, "airline", 12)     │                              │
 │─────────────────────────────>│                              │
 │                              │ load_dataset("airline")      │
 │                              │ fit(y, fh=[1..12])           │
 │                              │ predict(fh=[1..12])          │
 │<── {predictions: {...}} ─────│                              │
```

### Workflow 2: Custom Data → Format → Predict

```
LLM                              MCP Server
 │                                     │
 │ load_data_source({                  │
 │   type: "file",                     │
 │   path: "/data/sales.csv",          │
 │   time_column: "date",              │
 │   target_column: "revenue"          │
 │ }) ────────────────────────────────>│
 │                                     │ FileAdapter.load()
 │                                     │ FileAdapter.validate()
 │                                     │ to_sktime_format() → (y, X)
 │                                     │ auto-format if enabled
 │<── {data_handle: "data_abc"} ───────│
 │                                     │
 │ fit_predict_with_data(              │
 │   est_handle, "data_abc", 6)        │
 │────────────────────────────────────>│
 │                                     │ Executor.fit_predict_with_data()
 │<── {predictions: {...}} ────────────│
```

### Workflow 3: Background Job

```
LLM                              MCP Server                  Thread Pool
 │                                     │                          │
 │ fit_predict_async(                  │                          │
 │   handle, "airline", 12)            │                          │
 │────────────────────────────────────>│                          │
 │                                     │ JobManager.create_job()  │
 │                                     │ schedule coroutine ────> │
 │<── {job_id: "uuid-xxx"} ───────────│                          │
 │                                     │                          │
 │ check_job_status("uuid-xxx")        │                 (fitting)│
 │────────────────────────────────────>│                          │
 │<── {progress: 66%, step: "Fit..."} ─│                          │
 │                                     │                          │
 │ check_job_status("uuid-xxx")        │              (completed) │
 │────────────────────────────────────>│                          │
 │<── {status: "completed",            │                          │
 │     result: {predictions: ...}}─────│                          │
```

---

## 6. Tool Combination Strategies

### Recommended Multi-Tool Workflows

#### 1. Smart Forecasting Pipeline
```
get_available_tags        → Understand what capabilities exist
list_estimators           → Find estimators with specific capabilities
describe_estimator (x2)   → Deep-dive on transformer + forecaster
validate_pipeline         → Check composition before committing
instantiate_pipeline      → Create the pipeline
fit_predict              → Train and predict
export_code              → Get reproducible Python code
```

#### 2. Data-Driven Custom Forecasting
```
list_data_sources         → See supported adapters
load_data_source          → Load user data (auto-formats)
list_available_data       → Verify data loaded correctly
format_time_series        → Additional formatting if needed
instantiate_estimator     → Create forecaster
fit_predict_with_data     → Predict on user data
save_model               → Persist the trained model
export_code              → Generate reproducible code
release_data_handle       → Free memory
release_handle            → Free memory
```

#### 3. Model Comparison / Benchmarking
```
instantiate_estimator (×N)  → Create multiple estimators
evaluate_estimator (×N)     → Cross-validate each on same dataset
list_handles               → Review all active handles
export_code (×N)           → Export best performer
cleanup: release_handle (×N)
```

#### 4. Long-Running Training with Monitoring
```
instantiate_estimator       → Create model
fit_predict_async           → Start background training
check_job_status (poll)     → Monitor progress
list_jobs                  → Overview of all jobs
cancel_job                 → Cancel if needed
cleanup_old_jobs           → Housekeeping
```

#### 5. Model Persistence & Reuse
```
load_model                  → Load previously saved model
list_handles               → Verify it's loaded and fitted
fit_predict_with_data       → Use on new data
save_model                 → Save updated model
```

### Tool Synergy Improvements (Not Yet Implemented)

| Combination | Description |
|------------|-------------|
| `auto_benchmark` | Combine `list_estimators` + `instantiate` + `evaluate` in a single tool to benchmark top-N estimators automatically |
| `suggest_pipeline` | Use `CompositionValidator.suggest_pipeline()` (exists but not MCP-exposed) to auto-suggest pipelines for a task |
| `compare_models` | New tool that takes multiple handles and a dataset, runs evaluation on all, and returns a ranked comparison table |
| `data_profile` | New tool that analyzes a data handle and returns: seasonality detection, stationarity test, missing value report, recommended preprocessing steps |
| `auto_forecast` | End-to-end tool: load data → auto-detect characteristics → select estimator → fit → predict → export code |

---

## 7. Current Test Coverage & Testing Strategy

### Current State

**Test files:** 7 modules, 71 tests, **70 passing, 1 failing**

| Test File | Tests | Status | Coverage Area |
|-----------|-------|--------|---------------|
| `test_core.py` | 12 | All pass | Registry, HandleManager, CompositionValidator, basic tools |
| `test_codegen.py` | 17 | All pass | Value formatting, single estimator codegen, pipeline codegen, export_code |
| `test_param_validation.py` | 12 | All pass | `_validate_params`, `_is_safe_value`, pipeline params validation |
| `test_background_jobs.py` | 6 | All pass | Job creation, updates, listing, async fit_predict, cancel, cleanup |
| `test_async_data_loading.py` | 5 | All pass | Async data loading tool & executor |
| `test_evaluate.py` | 1 | **FAILS** | `evaluate_estimator_tool` — assertion mismatch (see §9) |
| `test_data_sources.py` | 18* | All pass | Data adapter imports, PandasAdapter, Executor integration |

*`test_data_sources.py` uses `print`-based assertions rather than `pytest.assert`, so failures would be silent.

### Coverage Gaps

The following are **not tested**:

1. **server.py dispatch** — No integration test calls `call_tool()` directly
2. **`sanitize_for_json()`** — No unit tests for Period, numpy, or nested object serialization
3. **`format_time_series_tool`** — No tests for frequency inference, gap filling, duplicate removal
4. **`save_model_tool` with real MLflow** — Only tested with monkeypatched mock
5. **`load_model_tool`** — No tests
6. **`export_code` with `include_fit_example=True` and custom dataset** — Partial coverage
7. **`UrlAdapter`** — No tests (would need network mocking)
8. **`SQLAdapter`** — No tests (would need database fixtures)
9. **Edge cases:** Empty datasets, very large datasets, non-datetime indices, multivariate time series
10. **Error paths:** Most error branches in tools are untested
11. **Concurrent access** — JobManager thread safety not stress-tested
12. **Handle eviction** — HandleManager's `_cleanup_oldest()` not tested

### Recommended Testing Strategy

#### Unit Tests to Add

```python
# tests/test_server_dispatch.py — Integration tests for server.py
class TestCallTool:
    """Test the call_tool dispatcher."""
    
    async def test_unknown_tool_returns_error(self):
        result = await call_tool("nonexistent_tool", {})
        assert "error" in json.loads(result[0].text)
    
    async def test_list_estimators_dispatches_correctly(self):
        result = await call_tool("list_estimators", {"limit": 5})
        parsed = json.loads(result[0].text)
        assert parsed["success"]

# tests/test_sanitize.py — Serialization edge cases
class TestSanitizeForJson:
    def test_period_index(self):
        ...
    def test_numpy_types(self):
        ...
    def test_nested_objects(self):
        ...

# tests/test_format_tools.py — Formatting logic
class TestFormatTimeSeries:
    def test_duplicate_removal(self):
        ...
    def test_frequency_inference_daily(self):
        ...
    def test_gap_filling(self):
        ...
    def test_missing_value_fill(self):
        ...

# tests/test_handle_manager.py — Edge cases
class TestHandleEviction:
    def test_evicts_oldest_when_full(self):
        ...
    def test_max_handles_boundary(self):
        ...

# tests/test_url_adapter.py — With mocked network
class TestUrlAdapter:
    @patch('urllib.request.urlretrieve')
    def test_downloads_and_loads_csv(self, mock_retrieve):
        ...
```

#### Integration Tests

```python
# tests/test_workflows.py — End-to-end MCP workflows
class TestForecastingWorkflow:
    """Test the complete discovery → instantiate → fit → predict flow."""
    
    def test_simple_forecast(self):
        result = list_estimators_tool(task="forecasting", limit=1)
        name = result["estimators"][0]["name"]
        inst = instantiate_estimator_tool(name)
        pred = fit_predict_tool(inst["handle"], "airline", 12)
        assert pred["success"]
        code = export_code_tool(inst["handle"])
        assert code["success"]

# tests/test_data_pipeline.py — Data loading workflows
class TestDataPipeline:
    def test_csv_load_format_predict(self, tmp_path):
        # Create CSV → load → format → predict → release
        ...
```

#### How to Run Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=sktime_mcp --cov-report=html

# Run specific module
pytest tests/test_core.py -v

# Run async tests only
pytest tests/ -v -k "async"
```

#### Recommended `pyproject.toml` additions:

```toml
[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
addopts = "--tb=short -q"
markers = [
    "slow: marks tests as slow (deselect with '-m \"not slow\"')",
    "integration: marks integration tests",
    "network: marks tests requiring network access",
]

[tool.coverage.run]
source = ["src/sktime_mcp"]
omit = ["tests/*"]

[tool.coverage.report]
fail_under = 70
show_missing = true
```

---

## 8. Documentation Review

### Current Documentation Files

| File | Lines | Quality | Issues |
|------|-------|---------|--------|
| `README.md` | ~530 | Good | Lists 17/28 tools; Python version contradiction (3.9+ vs 3.10+) |
| `docs/index.md` | ~105 | Good | Doc map table omits `background-jobs.md` |
| `docs/user-guide.md` | ~257 | Good | Tool table incomplete; some placeholder text |
| `docs/use-cases.md` | ~71 | Fair | URL loading marked "planned" but is implemented |
| `docs/usage-examples.md` | ~188 | Good | `serialization_format` inconsistency (pickle vs cloudpickle) |
| `docs/data-sources.md` | ~355 | Excellent | Most thorough doc; covers all data workflows |
| `docs/background-jobs.md` | ~408 | Excellent | Comprehensive job management documentation |
| `docs/architecture.md` | ~147 | Good | References images in `docs/assets/` that may not exist |
| `docs/implementation.md` | ~591 | Fair | Duplicated `pyproject.toml` section; incomplete server tool list; outdated examples list |
| `docs/dev-guide.md` | ~122 | Good | Missing `mkdocs-material` from dev dependencies |

### Key Documentation Issues

1. **Tool surface drift:** README and implementation.md only document ~17 of 28 registered tools. Missing: `evaluate_estimator`, `load_model`, `list_handles`, `release_data_handle`, `load_data_source_async`, `list_data_sources`, `cancel_job`, `delete_job`, `cleanup_old_jobs`, `auto_format_on_load`.

2. **Python version contradiction:** README says both "3.10+" and "3.9+". `pyproject.toml` says `>=3.9` but `mcp` SDK requires 3.10+. Should standardize on **3.10+**.

3. **Formatter mismatch:** Makefile uses Black; CI uses `ruff format`. They can disagree on formatting.

4. **use-cases.md:** States URL loading is "planned" — it's already implemented as `UrlAdapter`.

5. **PR template:** Copied from sktime core repo; links point to wrong repository. Also misplaced under `.github/workflows/` instead of `.github/`.

6. **mkdocs-material:** Required by `mkdocs.yml` but not in dev dependencies. `pip install -e ".[dev]"` won't install it.

7. **`site/` directory:** Built docs are committed to the repo. Should be in `.gitignore`.

### Recommended Documentation Improvements

- Create a single tool reference page listing all 28 tools with parameters and examples
- Standardize Python version to 3.10+ everywhere
- Update use-cases.md to reflect URL loading as implemented
- Fix PR template content and location
- Add `mkdocs-material` to dev dependencies
- Add `site/` to `.gitignore`
- Add CHANGELOG.md for release tracking

---

## 9. Bugs & Issues Found

### Critical

| # | Issue | Location | Impact |
|---|-------|----------|--------|
| 1 | **`auto_format_on_load` with omitted `enabled`** | `server.py:686` | `arguments["enabled"]` raises `KeyError` when client omits the param (it's not `required` in schema). Should use `arguments.get("enabled", True)`. |
| 2 | **Evaluate test assertion mismatch** | `test_evaluate.py:28` | `cv_folds_run == 2` fails because `ExpandingWindowSplitter` with the calculated `initial_window` produces 4 splits, not 2. `cv_folds` doesn't directly control fold count. |
| 3 | **`list_estimators` offset not wired** | `server.py:604-608` | `list_estimators_tool` accepts `offset` but `call_tool` never passes it; MCP schema omits it. |

### Moderate

| # | Issue | Location | Impact |
|---|-------|----------|--------|
| 4 | **`asyncio.get_event_loop()` deprecation** | `data_tools.py:189`, `executor.py:289` | Deprecated in Python 3.10+. Should use `asyncio.get_running_loop()`. |
| 5 | **`cancel_job` doesn't stop work** | `jobs.py:271-293` | Flips status to CANCELLED but the running coroutine/thread continues. Can lead to race conditions where work completes after cancellation. |
| 6 | **Data handles have no cap** | `executor.py:532` | Unlike estimator handles (max 100), data handles grow unbounded. Memory leak risk in long sessions. |
| 7 | **`ExpandingWindowSplitter` deprecated import** | `evaluate.py:11` | Import from `sktime.forecasting.model_selection` is deprecated; should use `sktime.split`. |
| 8 | **Singleton state leakage between tests** | Multiple singletons | `get_executor()`, `get_handle_manager()`, `get_job_manager()` persist across test functions, causing hidden state dependencies. |

### Minor

| # | Issue | Location | Impact |
|---|-------|----------|--------|
| 9 | **PR template misplaced** | `.github/workflows/` | GitHub won't auto-detect it. Should be at `.github/PULL_REQUEST_TEMPLATE.md`. |
| 10 | **Double sanitization** | `server.py:634,641,674,720` | `fit_predict` and `evaluate` results are sanitized twice (once explicitly, once in the final pass). |
| 11 | **`cv_folds` misleading name** | `evaluate.py` | Parameter is passed to `ExpandingWindowSplitter` but doesn't directly equal fold count. Should be renamed or documented. |
| 12 | **`test_data_sources.py` uses print assertions** | `tests/test_data_sources.py` | Failures are silent; should use `assert` statements. |
| 13 | **`site/` committed to repo** | `.gitignore` | Built docs shouldn't be in version control. |

---

## 10. Performance Analysis & Improvements

### Current Performance Characteristics

| Component | Bottleneck | Current Behavior |
|-----------|-----------|-----------------|
| **Registry loading** | First tool call | Loads ALL estimator types sequentially; can take 2-5 seconds depending on sktime installation |
| **Estimator instantiation** | Per-call | Instantiates fresh; no caching of commonly used estimators |
| **Dataset loading** | Per-call | Demo datasets are re-loaded on every `fit_predict` call |
| **Serialization** | Per-response | `sanitize_for_json` walks entire result recursively; then `json.dumps` with `default=str` double-walks |
| **Tag resolution** | First tag query | Loads all tags from sktime registry; cached thereafter |
| **Background jobs** | Thread pool | Uses default asyncio thread pool executor (usually small) |

### Recommended Performance Improvements

#### 1. Registry Pre-warming
```python
# Warm the registry at startup rather than on first tool call
async def run_server():
    # Pre-warm in background
    import asyncio
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, get_registry)
    
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, ...)
```

#### 2. Dataset Caching
```python
class Executor:
    def __init__(self):
        self._dataset_cache: dict[str, dict] = {}
    
    def load_dataset(self, name: str) -> dict:
        if name in self._dataset_cache:
            return self._dataset_cache[name]
        result = self._load_dataset_impl(name)
        if result["success"]:
            self._dataset_cache[name] = result
        return result
```

#### 3. Lazy Serialization
```python
# Avoid double-sanitization
if name in ("fit_predict", "evaluate_estimator", "fit_predict_with_data"):
    sanitized_result = sanitize_for_json(result)
else:
    sanitized_result = result  # Let json.dumps(default=str) handle simple cases
```

#### 4. Dedicated Thread Pool for Background Jobs
```python
import concurrent.futures

class Executor:
    def __init__(self):
        self._thread_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=4, thread_name_prefix="sktime-mcp-worker"
        )
```

#### 5. Response Size Limits
```python
def truncate_predictions(predictions: dict, max_items: int = 100) -> dict:
    """Truncate large prediction results for LLM consumption."""
    if len(predictions) > max_items:
        keys = list(predictions.keys())[:max_items]
        return {k: predictions[k] for k in keys}
    return predictions
```

#### 6. Tool Schema Generation
Replace hand-written JSON schemas with auto-generation from function signatures:
```python
from inspect import signature

def generate_schema(func) -> dict:
    sig = signature(func)
    properties = {}
    required = []
    for name, param in sig.parameters.items():
        if param.annotation != inspect.Parameter.empty:
            properties[name] = annotation_to_json_schema(param.annotation)
        if param.default is inspect.Parameter.empty:
            required.append(name)
    return {"type": "object", "properties": properties, "required": required}
```

#### 7. Connection Pooling for SQL Adapter
```python
from sqlalchemy import create_engine
from sqlalchemy.pool import QueuePool

class SQLAdapter(DataSourceAdapter):
    _engine_cache: dict[str, Engine] = {}
    
    def _get_engine(self, connection_string: str) -> Engine:
        if connection_string not in self._engine_cache:
            self._engine_cache[connection_string] = create_engine(
                connection_string, poolclass=QueuePool, pool_size=5
            )
        return self._engine_cache[connection_string]
```

---

## 11. Improvement Roadmap

### Phase 1: Bug Fixes & Stability (Immediate)

- [ ] Fix `auto_format_on_load` KeyError (use `arguments.get()`)
- [ ] Fix `test_evaluate.py` assertion to match actual behavior
- [ ] Update deprecated `ExpandingWindowSplitter` import
- [ ] Update deprecated `asyncio.get_event_loop()` calls
- [ ] Add cap on data handles (like estimator handles)
- [ ] Move PR template to `.github/PULL_REQUEST_TEMPLATE.md`
- [ ] Add `site/` to `.gitignore`
- [ ] Standardize Python version to 3.10+ in all docs

### Phase 2: Testing & Quality (Short-term)

- [ ] Add `pytest-cov` and achieve >70% coverage
- [ ] Convert `test_data_sources.py` to proper pytest assertions
- [ ] Add integration tests for `call_tool()` dispatcher
- [ ] Add unit tests for `sanitize_for_json()`
- [ ] Add unit tests for `format_time_series_tool`
- [ ] Add tests for handle eviction
- [ ] Fix singleton state leakage in tests (use fixtures with fresh instances)
- [ ] Add `conftest.py` with shared fixtures

### Phase 3: Performance & Architecture (Medium-term)

- [ ] Pre-warm registry at server startup
- [ ] Cache demo datasets
- [ ] Eliminate double serialization
- [ ] Dedicated thread pool for background jobs
- [ ] Auto-generate tool schemas from function signatures
- [ ] Refactor `server.py` dispatch from if/elif to a registry pattern
- [ ] Add `mkdocs-material` to dev dependencies

### Phase 4: Features & Tool Synergies (Long-term)

- [ ] New tool: `auto_benchmark` — auto-compare top-N estimators on a dataset
- [ ] New tool: `suggest_pipeline` — expose `CompositionValidator.suggest_pipeline()` as MCP tool
- [ ] New tool: `data_profile` — analyze data handle (seasonality, stationarity, missing values)
- [ ] New tool: `compare_models` — side-by-side comparison of multiple handles
- [ ] Expose `get_available_tasks` as MCP tool
- [ ] Expose `fit_tool` / `predict_tool` separately for fine-grained control
- [ ] Add `data_handle` support to `fit_predict_async`
- [ ] Implement true job cancellation (cancel underlying asyncio task)
- [ ] Add streaming progress updates via MCP notifications
- [ ] Add model versioning / history tracking
- [ ] Add support for classification/regression workflows end-to-end (not just forecasting)

### Phase 5: Deployment & Production (Long-term)

- [ ] Add Dockerfile for containerized deployment
- [ ] Add SSE/WebSocket transport option (beyond stdio)
- [ ] Add authentication/authorization layer
- [ ] Add rate limiting
- [ ] Add URL allowlist for `UrlAdapter` (SSRF prevention)
- [ ] Add configurable logging (file output, log levels)
- [ ] Add metrics/telemetry endpoint
- [ ] Publish to PyPI with CI/CD pipeline

---

## Appendix: File-by-File Reference

### Source Files (`src/sktime_mcp/`)

| File | Lines | Purpose |
|------|-------|---------|
| `__init__.py` | ~5 | Package marker |
| `server.py` | 742 | MCP server: tool registration, dispatch, serialization |
| `tools/__init__.py` | ~20 | Re-exports subset of tools |
| `tools/list_estimators.py` | ~80 | `list_estimators_tool`, `get_available_tags`, `get_available_tasks` |
| `tools/describe_estimator.py` | ~120 | `describe_estimator_tool`, `search_estimators_tool` |
| `tools/instantiate.py` | ~300 | Estimator/pipeline instantiation, handle management, model loading |
| `tools/fit_predict.py` | ~120 | `fit_predict_tool`, `fit_predict_async_tool` |
| `tools/evaluate.py` | ~74 | `evaluate_estimator_tool` |
| `tools/data_tools.py` | ~206 | Data loading, listing, async loading |
| `tools/format_tools.py` | ~95 | Time series formatting, auto-format toggle |
| `tools/codegen.py` | ~279 | Python code generation for estimators/pipelines |
| `tools/job_tools.py` | ~100 | Job management tool wrappers |
| `tools/save_model.py` | ~61 | Model persistence via MLflow |
| `tools/list_available_data.py` | ~50 | Unified data listing |
| `registry/__init__.py` | ~5 | Package marker |
| `registry/interface.py` | 344 | `RegistryInterface`, `EstimatorNode` |
| `registry/tag_resolver.py` | 297 | `TagResolver`, `TagInfo` |
| `runtime/__init__.py` | ~5 | Package marker |
| `runtime/executor.py` | 912 | Core execution engine |
| `runtime/handles.py` | 121 | Handle manager with LRU eviction |
| `runtime/jobs.py` | 342 | Thread-safe job manager |
| `composition/__init__.py` | ~5 | Package marker |
| `composition/validator.py` | 411 | Pipeline composition validator |
| `data/__init__.py` | ~30 | Data registry + adapter exports |
| `data/base.py` | 122 | Abstract adapter base class |
| `data/registry.py` | ~60 | `DataSourceRegistry` |
| `data/adapters/__init__.py` | ~10 | Adapter exports |
| `data/adapters/pandas_adapter.py` | ~180 | Pandas/dict data adapter |
| `data/adapters/file_adapter.py` | 220 | CSV/Excel/Parquet file adapter |
| `data/adapters/sql_adapter.py` | ~200 | SQL database adapter |
| `data/adapters/url_adapter.py` | 97 | URL download adapter |

### Test Files (`tests/`)

| File | Tests | Coverage |
|------|-------|----------|
| `test_core.py` | 12 | Registry, handles, composition, basic tools |
| `test_codegen.py` | 17 | Code generation |
| `test_param_validation.py` | 12 | Parameter validation |
| `test_background_jobs.py` | 6 | Job management |
| `test_async_data_loading.py` | 5 | Async data loading |
| `test_evaluate.py` | 1 | Evaluation (failing) |
| `test_data_sources.py` | ~18 | Data sources (print-based) |

### Example Files (`examples/`)

| File | Demonstrates |
|------|-------------|
| `01_forecasting_workflow.py` | Basic forecasting pipeline |
| `02_llm_query_simulation.py` | LLM-style query simulation |
| `03_pipeline_instantiation.py` | Pipeline creation |
| `04_mcp_pipeline_demo.py` | MCP pipeline workflow |
| `05_simple_deseasonalize_detrend_forecaster.py` | Deseasonalize + detrend |
| `06_simple_naive_forecaster.py` | NaiveForecaster example |
| `background_training_example.py` | Background job demo |
| `csv_example.py` | CSV data loading |
| `job_management_demo.py` | Job management |
| `pandas_example.py` | Pandas data loading |
| `sql_example.py` | SQL data loading |
