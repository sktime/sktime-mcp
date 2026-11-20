# Architecture

`sktime-mcp` is a specialized implementation of the Model Context Protocol (MCP) server designed to expose the `sktime` library's capabilities to Large Language Models.

## High-Level Overview

The server operates as a bridge between the stateless nature of LLM interactions and the stateful requirements of machine learning workflows. It manages object lifecycles, provides semantic discovery of estimators, and handles complex data ingestion.

```{image} ../_static/mcp_data_flow.png
:alt: Layered architecture of sktime-mcp, from user and LLM down through the MCP protocol layer, tools layer, executor layer, and data source layer.
:align: center
:width: 70%
```

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
`instantiate` spec string and validated by sktime's own `craft()`, so the server
carries no pipeline-validation code of its own.

### 5. Data Layer (`data/`)
Loading is adapter-based. `DataSourceRegistry` selects an adapter from the
`type` key of the incoming config:

```{image} ../_static/data_adapter_pattern.png
:alt: DataSourceAdapter abstract base class with concrete PandasAdapter, SQLAdapter, and FileAdapter subclasses.
:align: center
:width: 55%
```

Four adapters ship today: `PandasAdapter`, `SQLAdapter`, `FileAdapter`, and
`URLAdapter` (the diagram above predates the URL adapter).

## Data Flow: The "Handle" Pattern

1. **Instantiation**: The client requests a model (e.g., `ARIMA`) via `instantiate`. The server crafts the object, generates a UUID-based handle, and stores it in the `Handle Registry`.
2. **Action**: The client calls `fit`, then `predict`, passing the handle. The server retrieves the object from the registry and executes the requested method.
3. **Response**: The server serializes the result (e.g., forecast values) into JSON and returns it to the client.

The two flows below show this end to end — loading data to obtain a handle, and
then forecasting against it:

```{image} ../_static/component_interaction.png
:alt: Sequence diagrams for the data loading flow and the forecasting flow, showing user, MCP server, executor, adapter, and estimator interactions.
:align: center
:width: 90%
```

## Concurrency Model

- **Sync tools**: Run on the main event loop. These should be fast, metadata-only operations.
- **Async tools**: `fit`, `predict`, `evaluate`, and `load_data_source` accept a `run_async` flag. When set, the work is scheduled with `asyncio.create_task`, the task reference is retained by the `JobManager`, and a `job_id` is returned immediately. This keeps a long training run from blocking the server, so the client can keep querying status or metadata.
- Blocking sktime calls that would otherwise stall the loop are offloaded with `run_in_executor`.
- Because task references are retained, `cancel_job` can actually cancel in-flight work. Cancellation is cooperative: a step already inside a blocking sktime call completes before the task stops.

## Security Considerations

The trust boundary is **the user account running the server**. There is no
authentication layer and no sandbox.

- **Strict typing**: All tool inputs are validated against JSON schemas.
- **Registry-bounded modelling**: The modelling tools (`instantiate`, `fit`, `predict`, …) only construct classes resolvable through the `sktime` registry via `craft()`. They do not evaluate caller-supplied Python.
- **`run_command` is an explicit exception.** The server also exposes a `run_command` tool that executes an arbitrary shell command with the server process's full privileges, so that an assistant can install a missing optional dependency mid-session. It is not sandboxed. Any client permitted to call it can run anything the user can. Treat it as opt-in: most MCP clients can allow or deny individual tools, and `run_command` should stay on ask-every-time — or be disabled — unless you want that capability. Running the server in a container confines it to the container.
- **Local access**: The server inherits the permissions of the user running it and can read/write any file that user can.

See the {doc}`../tool-reference` for the per-tool detail.
