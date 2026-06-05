# Architecture

`sktime-mcp` is a specialized implementation of the Model Context Protocol (MCP) server designed to expose the `sktime` library's capabilities to Large Language Models.

## High-Level Overview

The server operates as a bridge between the stateless nature of LLM interactions and the stateful requirements of machine learning workflows. It manages object lifecycles, provides semantic discovery of estimators, and handles complex data ingestion.

```text
graph TD
    Client[MCP Client: Claude/Cursor] <--> Server[sktime-mcp Server]
    Server <--> Registry[sktime Estimator Registry]
    Server <--> Runtime[Stateful Runtime: Handles & Jobs]
    Runtime <--> Objects[Estimators & Datasets in RAM]
    Runtime <--> Storage[Local Filesystem / MLflow]
```

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
- **Data**: Ingesting and formatting time series.
- **Execution**: Fitting, predicting, and evaluating.
- **Composition**: Building pipelines.
- **Persistence**: Saving models and exporting code.

## Data Flow: The "Handle" Pattern

1. **Instantiation**: The client requests a model (e.g., `ARIMA`). The server instantiates the class, generates a UUID-based handle, and stores the object in the `Handle Registry`.
2. **Action**: The client requests an action (e.g., `fit_predict`) providing the handle. The server retrieves the object from the registry and executes the requested method.
3. **Response**: The server serializes the result (e.g., forecast values) into JSON and returns it to the client.

## Concurrency Model

- **Sync Tools**: Run on the main event loop. These should be fast, metadata-only operations.
- **Async Tools**: Heavy operations (like `fit_predict_async`) are dispatched to a `ThreadPoolExecutor`. This prevents a single long-running training job from blocking the entire server, allowing the client to continue querying status or other metadata.

## Security Considerations

- **Strict Typing**: All tool inputs are validated against JSON schemas.
- **Restricted Execution**: The server only allows instantiation of classes found in the `sktime` registry. It does not permit arbitrary code execution.
- **Local Access**: The server has the same permissions as the user running it. It can read/write files based on the user's filesystem access.
