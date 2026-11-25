# Installation

Use Python **3.10 or newer**. Install only the extras needed by your workflow.
The registry's dependency hints describe additional estimator requirements.

## Current source checkout

The new 25-tool interface is an unreleased source change. To use these exact docs,
install the checkout containing the changes, rather than assuming PyPI 0.1.3 has them:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
python -m sktime_mcp.server --version
```

On Windows, activate with `.venv\Scripts\Activate.ps1`. Point the MCP client at
that environment's Python executable. Record `git rev-parse HEAD` and whether the
working tree has changes when reproducing a development workflow.

## Published package

For the published interface:

```bash
python -m pip install sktime-mcp
```

With uv, `uvx sktime-mcp` launches the published package in an isolated environment.
Published and development tool names differ; see {doc}`migration`.

## Optional dependencies

| Extra | Adds |
| --- | --- |
| `files` | Excel readers and Parquet support |
| `sql` | SQLAlchemy, PostgreSQL and MySQL drivers |
| `forecasting` | Prophet, TBATS and StatsForecast |
| `mlflow` | Compatibility extra; MLflow is already a core dependency |
| `dl` | TensorFlow and PyTorch integrations |
| `dev` | Tests, linting and documentation tools |
| `all` | The bundled file, SQL and forecasting extras; excludes `dl` and many optional sktime dependencies |

For example, install Excel/Parquet and SQL support in this checkout with:

```bash
python -m pip install -e ".[files,sql]"
```

There is no need to install deep-learning dependencies for the examples in this guide.

## Docker

Build from the checkout you want to run:

```bash
docker build -t sktime-mcp .
docker run --rm -i sktime-mcp
```

Mount only the directories the workflow needs. Container file paths differ from
host paths; configure `load_data` and `save_data` accordingly.

## Verify the environment

`python -m sktime_mcp.server --help` should print the CLI options. Starting without
arguments opens the stdio server and waits for MCP messages; an idle terminal is
expected. In the client, inspect the tool list: this checkout advertises `load_data`
and `prepare_data`, with **25 tools total**.
