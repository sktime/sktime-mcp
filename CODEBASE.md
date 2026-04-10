# sktime-mcp Codebase Overview

## 1. PROJECT OVERVIEW

sktime-mcp is a Python package that implements an MCP (Model Context Protocol) layer for sktime. It exposes sktime estimator discovery, description, instantiation, composition, execution, and persistence to LLM clients via an MCP server.

The project is a semantic registry layer rather than a UI app. It focuses on:
- loading sktime estimators from the sktime registry
- exposing estimator metadata, tags, and hyperparameters
- validating pipeline composition rules
- instantiating estimators and pipelines
- fitting and predicting on demo or loaded data
- managing in-memory handles and background jobs

Tech stack and versions from `pyproject.toml`:
- Python: `>=3.9`
- Build: `hatchling`
- Core runtime: `mcp>=1.0.0`, `sktime>=0.24.0`, `pandas>=1.5.0`, `numpy>=1.21.0`, `scikit-learn>=1.0.0`, `statsmodels>=0.13.0`, `pmdarima>=2.0.0`
- Optional dependencies:
  - dev: `pytest>=7.0.0`, `pytest-asyncio>=0.21.0`, `black>=23.0.0`, `ruff>=0.1.0`, `mkdocs>=1.5.0`
  - forecasting: `prophet>=1.1.0`, `tbats>=1.1.0`, `statsforecast>=1.4.0`
  - dl: `tensorflow>=2.9.0`, `torch>=1.9.0`
  - sql: `sqlalchemy>=2.0.0`, `psycopg2-binary>=2.9.0`, `pymysql>=1.0.0`
  - files: `openpyxl>=3.0.0`, `pyarrow>=10.0.0`

## 2. FOLDER STRUCTURE

### `src/sktime_mcp/`
- `__init__.py` - exports the public package API and key classes.
- `server.py` - MCP server entry point, tool registration, stdio transport, and tool descriptions.

### `src/sktime_mcp/composition/`
- `__init__.py` - package marker.
- `validator.py` - enforces valid estimator compositions and pipeline rules.

### `src/sktime_mcp/data/`
- `__init__.py` - package marker.
- `registry.py` - central registry of data source adapters.
- `base.py` - abstract base class for data source adapters.

### `src/sktime_mcp/data/adapters/`
- `__init__.py` - package marker.
- `file_adapter.py` - loads CSV/Excel/Parquet files and converts them to sktime format.
- `pandas_adapter.py` - loads data from pandas DataFrames and validates them.
- `sql_adapter.py` - loads SQL query results via SQLAlchemy and converts them to pandas.
- `url_adapter.py` - loads data from URL endpoints and converts to pandas.

### `src/sktime_mcp/registry/`
- `__init__.py` - package marker.
- `interface.py` - wraps sktime `all_estimators`, builds `EstimatorNode`, filters by task/tags.
- `tag_resolver.py` - loads sktime tag definitions and explains estimator capability tags.

### `src/sktime_mcp/runtime/`
- `__init__.py` - exports runtime classes.
- `executor.py` - core runtime operations: instantiate, fit, predict, pipelines, data loading, async jobs, formatting, handle/data management.
- `handles.py` - in-memory handle manager for estimator instances.
- `jobs.py` - in-memory background job manager for async workflows.

### `src/sktime_mcp/tools/`
- `__init__.py` - package marker.
- `codegen.py` - code export tool (likely exports code snippets for workflows).
- `data_tools.py` - tool wrappers for loading data sources, listing adapters, and data-backed fit/predict.
- `describe_estimator.py` - tool for estimator descriptions and search.
- `fit_predict.py` - tool wrappers for fit/predict workflows and async training.
- `format_tools.py` - format tools for time series and data formatting.
- `instantiate.py` - tool wrappers for estimator/pipeline instantiation and handle management.
- `job_tools.py` - tools for job status, listing, canceling, deleting, and cleanup.
- `list_available_data.py` - tool for listing demo datasets and active data handles.
- `list_estimators.py` - tool for discovery of estimators, tasks, and tags.
- `save_model.py` - tool to save instantiated estimators via sktime MLflow integration.

## 3. DATA FLOW

### How data moves from user action to runtime
1. An LLM client calls the MCP server via stdio transport.
2. `server.py` receives the tool request and dispatches it to the matching tool function.
3. Tool wrappers in `src/sktime_mcp/tools/` validate inputs and forward calls to the runtime.
4. The runtime operates against:
   - `RegistryInterface` for estimator metadata
   - `HandleManager` for estimator instances
   - `JobManager` for background tasks
   - `DataSourceRegistry` for loading custom data

### Example: estimator creation and execution
- `instantiate_estimator_tool` validates `params` and calls `Executor.instantiate()`.
- `Executor.instantiate()` uses `RegistryInterface.get_estimator_by_name()`, constructs the estimator, and stores it in `HandleManager`.
- `fit_predict_tool` loads a demo dataset, fits the estimator instance, and returns predictions.

### Data adapter flow
- `load_data_source_tool` calls `Executor.load_data_source(config)`.
- `Executor.load_data_source()` creates an adapter via `DataSourceRegistry.create_adapter(config)`.
- The adapter loads raw data, validates it, converts it to sktime format, and returns a `data_handle`.
- `Executor` stores loaded data in `_data_handles`, optionally auto-formats it, and returns metadata.

### BookingModal / Supabase / EmailJS / Web3Forms
- This repository does not contain any UI components like `BookingModal`.
- There is no Supabase, EmailJS, Web3Forms, or Chatwoot integration in the codebase.
- All external integrations are Python libraries only.

## 4. COMPONENT MAP

This is not a frontend component tree. The main components are Python modules and runtime classes.

### Core package components
- `sktime_mcp.server` — MCP server, tool registration, JSON-safe serialization.
- `sktime_mcp.registry.interface.RegistryInterface` — loads estimator metadata from sktime.
- `sktime_mcp.registry.tag_resolver.TagResolver` — interprets sktime tags and capability definitions.
- `sktime_mcp.composition.validator.CompositionValidator` — validates proposed estimator pipelines.
- `sktime_mcp.runtime.executor.Executor` — executes instantiate/fit/predict/pipeline/data workflows.
- `sktime_mcp.runtime.handles.HandleManager` — stores active estimator handles.
- `sktime_mcp.runtime.jobs.JobManager` — tracks async background jobs.

### Tool components
- `describe_estimator_tool` — search and describe estimators.
- `list_estimators_tool` — filter estimators by task and tags.
- `instantiate_estimator_tool` — instantiate a single estimator.
- `instantiate_pipeline_tool` — build pipeline instances.
- `fit_predict_tool` / `fit_predict_async_tool` — execute forecasting workflows.
- `load_data_source_tool` — load arbitrary data sources.
- `list_available_data_tool` — list demo datasets plus active data handles.
- `save_model_tool` — persist models using sktime MLflow utils.
- `job_tools` — status, list, cancel, delete, cleanup.

### Lazy loading
- `RegistryInterface` lazily loads the sktime registry on first use.
- `TagResolver` lazily loads tag definitions from `sktime.registry.all_tags()`.
- `Executor.load_dataset()` dynamically imports demo dataset loaders.
- `save_model_tool` lazily imports `sktime.utils.mlflow_sktime.save_model`.

## 5. ROUTING

There is no HTTP or frontend routing in this repository.

- The only request routing is MCP tool dispatch in `server.py`.
- Each tool is exposed by name, not by URL path.
- There are no dynamic routes or route paths defined.

## 6. STATE MANAGEMENT

State is managed in-memory by singleton managers:

- `HandleManager` stores estimator instances and metadata keyed by handle IDs.
- `JobManager` stores background job state and progress.
- `Executor` stores loaded data handles in `_data_handles`.
- `RegistryInterface` caches loaded estimator metadata in `_cache` and available tags.
- `DataSourceRegistry` maintains the registry of adapter classes.

State flow examples:
- `instantiate_estimator` creates a handle in `HandleManager`.
- `fit_predict_async` creates a job in `JobManager`, updates progress, and returns a `job_id`.
- Data loading stores metadata and validation state in `Executor._data_handles`.

## 7. EXTERNAL INTEGRATIONS

### Supabase
- None. The codebase does not integrate with Supabase.

### EmailJS
- None. The codebase does not integrate with EmailJS.

### Web3Forms
- None. The codebase does not integrate with Web3Forms.

### Chatwoot
- None. The codebase does not integrate with Chatwoot.

### Actual external integrations in this repo
- `mcp` — provides MCP server and tool definitions.
- `sktime` — source of estimator registry, tags, transformers, and forecasters.
- `pandas` / `numpy` — data loading, validation, and conversion.
- `scikit-learn` — used indirectly by sktime pipelines and transformers.
- `statsmodels`, `pmdarima` — forecasting estimators and dependencies.
- Optional SQL support: `sqlalchemy`, `psycopg2-binary`, `pymysql`.
- Optional file support: `openpyxl`, `pyarrow`.
- Optional MLflow persistence via `sktime.utils.mlflow_sktime.save_model`.

## 8. KNOWN ISSUES

Identified risks and limitations from source analysis:

- `RegistryInterface._load_registry()` may silently skip estimator types if imports fail, so the registry can be incomplete without clear reporting.
- `CompositionValidator` only supports a limited set of pipeline patterns and may reject valid sktime compositions not covered by its rules.
- `Executor.instantiate_pipeline()` supports only:
  - transformer → forecaster via `TransformedTargetForecaster`
  - transformer → classifier/regressor via `Pipeline`
  - transformer-only chains via `TransformerPipeline`
  It may return "Unsupported pipeline composition type" for valid cases outside these patterns.
- Async job scheduling uses `asyncio.get_event_loop()` and `asyncio.run_coroutine_threadsafe()`, which can be fragile in some execution environments.
- `save_model_tool` requires optional MLflow support but does not declare MLflow in core dependencies.
- `format_data_handle()` can infer a frequency and reindex data automatically, which may modify the original data timeline and hide issues.
- There are no environment variable hooks or configuration files for server behavior.
- No evidence of authentication, access control, or persistence beyond process memory.

## 9. ENVIRONMENT VARIABLES

This repository does not define or require any `.env` variables in its source code.

- No `os.getenv`, `os.environ`, or dotenv usage was found in `src/`.
- The MCP server runs from the command `sktime-mcp` or `python -m sktime_mcp.server` without environment-specific configuration.

## ADDITIONAL NOTES

- The package entrypoint is `sktime-mcp`, defined in `pyproject.toml`.
- Documentation is expected in `docs/` and site generation in `site/`, but the primary runtime is under `src/`.
- This is a backend library/service, not a frontend application.
