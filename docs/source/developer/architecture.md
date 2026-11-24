# Architecture

`sktime-mcp` is a specialized implementation of the Model Context Protocol (MCP) server designed to expose the `sktime` library's capabilities to Large Language Models.

## High-Level Overview

The server operates as a bridge between the stateless nature of LLM interactions and the stateful requirements of machine learning workflows. It manages object lifecycles, provides semantic discovery of estimators, and handles complex data ingestion.

| Layer | Current implementation | Responsibility |
| :--- | :--- | :--- |
| MCP protocol | `server.py` | Advertise tool schemas and dispatch requests. |
| Tool functions | `tools/` | Validate tool-specific inputs and assemble responses. |
| Execution | `runtime/executor.py` | Resolve data and run sktime operations. |
| State | `runtime/handles.py`, `runtime/jobs.py` | Retain estimator handles and background jobs. |
| Discovery | `registry/` | Read registered classes, parameters and capability tags. |
| Data access | `data/registry.py`, `data/adapters/` | Select and run the pandas, SQL, file or URL adapter. |


A request enters at the MCP protocol layer (`server.py`), is dispatched to a tool
(`tools/`), which delegates the actual work to the executor (`runtime/executor.py`).
Data access is resolved by the data source layer (`data/registry.py`) against one of
the adapters — pandas, SQL, file, or URL.

## Core Components

### 1. Server Layer (`server.py`)
This layer implements the MCP specification. It defines the tool schemas and dispatches incoming requests to the appropriate internal modules. It also handles JSON serialization/deserialization, ensuring that complex Python types are safely converted for the LLM.

### 2. Registry Module (`registry/`)
The registry module dynamically inspects `sktime`. It uses `sktime.registry.all_estimators` to discover available models and their capabilities (tags). This ensures the server is always in sync with the installed version of `sktime`.

### 3. Runtime and Handle Registry (`runtime/`)
The runtime is the heart of the server's statefulness. It maintains two primary registries:
- **Handle Registry**: A mapping of string identifiers to active Python objects (estimators and datasets).
- **Job Registry**: Tracks the status and results of background operations.

### 4. Tool Implementation (`tools/`)
Tools are modularized into functional groups:
- **Discovery**: Querying the registry.
- **Data**: Ingesting, inspecting, splitting, formatting, and plotting time series.
- **Execution**: Instantiating, fitting, predicting, and evaluating.
- **Persistence**: Saving models and exporting code.
- **Jobs**: Tracking and cancelling background work.

Composition is **not** a separate layer. Pipelines are expressed inside the
`instantiate` spec string and constructed by sktime's own `craft()`. Compatibility errors can arise during
construction or later fitting; the server carries no independent proof of pipeline
compatibility.

### 5. Data Layer (`data/`)
Loading is adapter-based. `DataSourceRegistry` selects an adapter from the
`type` key of the incoming config:



Four adapters ship today: `PandasAdapter`, `SQLAdapter`, `FileAdapter`, and
`UrlAdapter`.

## Data Flow: The "Handle" Pattern

1. **Instantiation**: The client requests a model (e.g., `ARIMA`) via `instantiate`. The server crafts the object, generates a UUID-based handle, and stores it in the `Handle Registry`.
2. **Action**: The client calls `fit`, then `predict`, passing the handle. The server retrieves the object from the registry and executes the requested method.
3. **Response**: The server serializes the result (e.g., forecast values) into JSON and returns it to the client.

A data load uses `load_data_source`, then the executor selects an adapter and
retains the loaded target and optional exogenous data behind a data handle.
A forecasting call resolves that data handle plus an estimator handle, invokes the
sktime method, and returns the serialized result. `predict` also stores the full
forecast behind a `prediction_handle` for plotting, inspection or saving.

## Concurrency Model

- **Sync tools**: Run on the main event loop. These should be fast, metadata-only operations.
- **Async tools**: `fit`, `predict`, `evaluate`, and `load_data_source` accept a `run_async` flag. When set, the work is scheduled with `asyncio.create_task`, the task reference is retained by the `JobManager`, and a `job_id` is returned immediately. This keeps a long training run from blocking the server, so the client can keep querying status or metadata.
- Blocking sktime calls that would otherwise stall the loop are offloaded with `run_in_executor`.
- Because task references are retained, `cancel_job` can actually cancel in-flight work. Cancellation is cooperative: a step already inside a blocking sktime call completes before the task stops.

## Security Considerations

The execution boundary is **the user account running the server**, not a sandbox.

- **Transport access:** local stdio relies on the client and operating-system process permissions. HTTP/SSE requires `SKTIME_MCP_HTTP_TOKEN`, except for the public `/` health check. Startup without a token fails unless `SKTIME_MCP_HTTP_INSECURE=true` is explicitly enabled for local testing. Protected routes require `Authorization: Bearer <token>`; configured host/origin allowlists also apply.
- **Description lookup:** `describe_component` resolves registered classes and known qualified aliases without evaluating input or importing client-supplied paths.
- **Instantiation:** `instantiate` passes caller-supplied specifications to `craft()`, an expression evaluator. It checks that the result is an sktime object, but evaluation occurs before that result check. Restrict this tool to trusted specifications and clients.
- **Shell access:** `run_command` executes arbitrary shell commands with the process's privileges. Keep it disabled or subject to explicit client approval unless needed.
- **Input contracts:** tools advertise JSON schemas and perform tool-specific validation. These checks help reject invalid requests but do not provide code isolation.
- **Filesystem access:** tools operate on paths accessible to the server account. Container isolation depends on its mounts, privileges and configuration. HTTP authentication does not reduce an authorized client's execution privileges.

See the {doc}`../tool-reference` for per-tool behavior and the README for HTTP
configuration. The broader execution trust boundary is tracked in
[issue #540](https://github.com/sktime/sktime-mcp/issues/540).
