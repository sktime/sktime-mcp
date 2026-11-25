# sktime-mcp

A Model Context Protocol server for discovering and running sktime components.
Models, datasets and background jobs stay in a shared runtime so an assistant can
continue a workflow across tool calls.

[Documentation](https://sktime-mcp.readthedocs.io/en/latest/) ·
[PyPI](https://pypi.org/project/sktime-mcp/) ·
[Tool reference](docs/source/tool-reference.md) ·
[Support matrix](docs/source/support-matrix.md)

> **Development interface:** this checkout exposes 25 tools, including `load_data`
> and `prepare_data`. These changes are unreleased; the published package version
> remains 0.1.3. See the [migration guide](docs/source/migration.md) before updating
> an existing client.

## Install and connect

For this source checkout:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
python -m sktime_mcp.server --version
```

Point your MCP client at that environment. Clients using an `mcpServers` object
can use:

```json
{
  "mcpServers": {
    "sktime": {
      "command": "/absolute/path/to/.venv/bin/python",
      "args": ["-m", "sktime_mcp.server"]
    }
  }
}
```

For the published release, use `pip install sktime-mcp` or `uvx sktime-mcp`.
Optional extras include `files`, `sql`, `forecasting`, `dl`, and `dev`.
MLflow is already a core dependency; the `mlflow` extra repeats it for compatibility.
See [installation](docs/source/installation.md) for the dependencies your workflow needs.

To run a container built from this checkout:

```bash
docker build -t sktime-mcp .
docker run --rm -i sktime-mcp
```

## The 25 tools

| Workflow | Tools |
| --- | --- |
| Discover | `query_registry`, `describe_component`, `list_available_data` |
| Work with data | `load_data`, `inspect_data`, `prepare_data`, `save_data` |
| Create and run components | `instantiate`, `fit`, `predict`, `update`, `get_fitted_params`, `call_method` |
| Evaluate and visualize | `evaluate`, `plot_series` |
| Reproduce and persist | `export_code`, `save_model`, `load_model` |
| Manage handles | `list_handles`, `release_handle`, `release_data_handle` |
| Manage jobs | `list_jobs`, `check_job_status`, `cancel_job` |
| Run a command | `run_command` |

Loading, fitting, prediction and evaluation accept `run_async=true`; poll the
returned job ID. Handles are process-wide and disappear on restart or eviction.

## Data without silent repair

`load_data` reads memory/inline containers, files, SQL or HTTP(S) URLs. It uses
sktime's data contracts and preserves values, row order, missing observations,
duplicates and multiple variables. Declare time, target and feature roles explicitly.

`prepare_data` combines three operations:

- **format:** explicit sorting, duplicate handling, missing-value policy and bounded frequency expansion;
- **convert:** sktime mtype conversion within a scitype, preserving unselected X/y components;
- **split:** temporal holdouts per series or explicit whole-instance/Table-row holdouts.

Every preparation creates new handles and retains its source. Loads and preparation
results include bounded inspection and validation. Read the
[data guide](docs/source/data-guide.md) for Series, Panel, Hierarchical, Table,
Alignment and Proba representation support and operation-specific limits.

Forecasting has the broadest end-to-end coverage. Custom classification can use
feature panels and separate labels. `evaluate` remains Series-forecasting-specific;
plotting, export and native-method lifecycle support are not uniform across tasks.
The [support matrix](docs/source/support-matrix.md) records these boundaries.

## Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `SKTIME_MCP_LOG_LEVEL` | `WARNING` | Logging verbosity |
| `SKTIME_MCP_LOG_PATH` | unset | Optional log file |
| `SKTIME_MCP_URL_TIMEOUT_SECONDS` | `30` | Total HTTP(S) download deadline |
| `SKTIME_MCP_URL_MAX_BYTES` | `52428800` | 50 MiB downloaded/decoded size limit |
| `SKTIME_MCP_MAX_PREPARED_ROWS` | `1000000` | Frequency-expansion row limit |
| `SKTIME_MCP_MAX_DATA_HANDLES` | `50` | Retained data handles |
| `SKTIME_MCP_MAX_HORIZON` | `10000` | Forecast/holdout horizon limit |
| `SKTIME_MCP_MAX_RESPONSE_TOKENS` | `0` | Response budget; zero disables the cap |
| `SKTIME_MCP_JOB_MAX_AGE_HOURS` | `24` | Retention for finished jobs |
| `SKTIME_MCP_JOB_CLEANUP_INTERVAL` | `3600` | Cleanup interval in seconds |
| `SKTIME_MCP_HTTP_TOKEN` | unset | Required HTTP/SSE bearer token |
| `SKTIME_MCP_HTTP_INSECURE` | `false` | Explicit unauthenticated local-test mode |
| `SKTIME_MCP_HTTP_ALLOWED_HOSTS` | `127.0.0.1,127.0.0.1:*,localhost,localhost:*` | Accepted hosts |
| `SKTIME_MCP_HTTP_ALLOWED_ORIGINS` | unset | Allowed browser origins |

Integer settings are validated; invalid values fall back to their defaults.
`SKTIME_MCP_AUTO_FORMAT` is deprecated and ignored: preparation is explicit.
URL parsing is outside the download deadline. These settings are not universal
memory, SQL-query or estimator runtime limits.

## Execution boundary

Stdio runs under local process permissions. HTTP/SSE requires bearer authentication
unless insecure mode is explicitly enabled. The health endpoint is public.
Craft specifications and `run_command` execute with the server account's privileges;
they are not sandboxed. Use trusted clients and appropriate process/container access.

Cancelling a fit no longer blocks the server while its worker finishes. Python
cannot safely kill that running thread; the estimator may still change after
cancellation. Use a fresh estimator for subsequent work.

## Develop and verify

```bash
ruff check .
ruff format --check .
pytest -q
sphinx-build -W --keep-going docs/source docs/_build/html
```

Read [contributing](docs/source/developer/contributing.md) for source layout,
contract tests and browser review. The examples use small data and lightweight
models; deep-learning dependencies are unnecessary.
