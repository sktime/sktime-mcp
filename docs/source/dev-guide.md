# Developer Guide

This guide explains how the project is structured, how to develop new features, and how to test changes.

## Development Prerequisites

- Python 3.10+
- pip
- Optional: `mkdocs` if you want to build the documentation site

## Setup

From the repo root:

```bash
pip install -e ".[dev]"
```

Validate the environment before running tests:

```bash
python -c "import sktime; print(sktime.__version__)"
```

If this command fails with `ModuleNotFoundError: No module named 'sktime'`, reinstall project dependencies in the active virtual environment:

```bash
pip install -e ".[dev]"
```

If you work with SQL or file formats:

```bash
pip install -e ".[sql]"
pip install -e ".[files]"
```

## Running The Server Locally

```bash
python -m sktime_mcp.server
```

## Running Tests

```bash
pytest
```

## Standard Local Checks

From the repo root:

```bash
make check
```

Auto-fix formatting and fixable lint issues:

```bash
make format-fix
```

If `make` is unavailable (common on Windows), run:

```bash
python -m black --check .
python -m ruff check .
python -m pytest
```

## Project Layout

- `src/sktime_mcp/server.py` MCP entry point and tool router
- `src/sktime_mcp/tools/` MCP tool implementations
- `src/sktime_mcp/registry/` sktime estimator discovery and tag logic
- `src/sktime_mcp/composition/` pipeline validation rules
- `src/sktime_mcp/runtime/` executor and handle manager
- `src/sktime_mcp/data/` data source adapters and registry
- `examples/` runnable examples

## Architecture Overview

High-level flow:

1. MCP server receives a tool call
2. The server routes to a tool implementation in `tools/`
3. Tools use the registry/runtime layers to execute operations
4. Results are sanitized and returned to the client

This separation keeps the MCP surface small while allowing deeper functionality in the runtime and data layers.

## Adding A New MCP Tool

1. Create a new module under `src/sktime_mcp/tools/` with a `*_tool` function.
2. Add a tool schema in `src/sktime_mcp/server.py` within `list_tools()`.
3. Add a handler in `call_tool()` to dispatch the tool name.
4. Add tests in `tests/`.
5. Document the tool in `docs/user-guide.md`.

## Adding A New Data Source Adapter

1. Implement a new adapter in `src/sktime_mcp/data/adapters/` that inherits `DataSourceAdapter`.
2. Register it in `src/sktime_mcp/data/registry.py` using `DataSourceRegistry.register()`.
3. Add tests covering `load()`, `validate()`, and `to_sktime_format()`.
4. Update `docs/data-sources.md`.

## Adding A Demo Dataset

Update `DEMO_DATASETS` in `src/sktime_mcp/runtime/executor.py` with a new loader string. Ensure the loader function returns a sktime-compatible dataset.

## Documentation Build

If you want to build the docs site locally:

```bash
pip install mkdocs
mkdocs serve
```

The config lives in `mkdocs.yml` and the docs content is under `docs/`.
