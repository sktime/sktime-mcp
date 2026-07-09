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

### Zero-install via uvx (recommended)

If you have [uv](https://github.com/astral-sh/uv) installed, no separate installation step is needed. Just update your MCP client config (see [Connecting from an LLM Client](#connecting-from-an-llm-client) below) and `uvx` will handle the rest automatically.

```bash
# Verify uv is available
uvx sktime-mcp --help
```

### pip

```bash
pip install sktime-mcp

# With optional extras (SQL, forecasting models, file formats)
pip install "sktime-mcp[all]"
```

### Development installation

```bash
git clone https://github.com/sktime/sktime-mcp
cd sktime-mcp
python3 -m pip install -e ".[dev]"
```

### 🐳 Docker

Run without installing anything locally (only Docker required):

```bash
# Build the image
docker build -t sktime-mcp .

# Run the MCP server (stdio transport)
docker run -i sktime-mcp
```

Or use Docker Compose:

```bash
docker compose build
docker compose run sktime-mcp
```

**Claude Desktop** — use Docker as the MCP server command:

```json
{
  "mcpServers": {
    "sktime": {
      "command": "docker",
      "args": ["run", "-i", "--rm", "sktime-mcp"]
    }
  }
}
```

Environment variables can be passed at runtime:

```bash
docker run -i -e SKTIME_MCP_LOG_LEVEL=DEBUG sktime-mcp
```

For a more detailed first-time setup flow, including MCP server verification and troubleshooting, see [Beginner Setup](#-beginner-setup-firsttime-users).

## 🧭 Beginner Setup (First‑Time Users)

If you are new to sktime‑mcp or to MCP‑based workflows, this section provides a minimal starting point to help you verify that your setup is working correctly.

### What is MCP?
The Model Context Protocol (MCP) allows Large Language Models (LLMs) to discover, reason about, and execute sktime workflows programmatically. This project exposes sktime’s estimator registry and semantics in a structured way so that LLMs can safely compose and run real time‑series pipelines.

### Prerequisites
- Python 3.10 or newer
- A working Python virtual environment (recommended)
- `pip` installed

### macOS / Unix-like first-time setup

For macOS or Unix-like shells, create an isolated virtual environment before installing the package:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install sktime-mcp
```

For development (if you want to modify the source):

```bash
python -m pip install -e ".[dev]"
```

**Verify that the MCP server starts:**

```bash
sktime-mcp
```

If the `sktime-mcp` console command is not found (e.g. the script was not placed on your `PATH`), use the module fallback instead — this is also the recommended form when an MCP client needs to target a specific Python environment:

```bash
python -m sktime_mcp.server
```

**Common first-time issues:**

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `command not found: sktime-mcp` | Scripts directory not on `PATH` | Run `python -m sktime_mcp.server` or add `.venv/bin` to your `PATH` |
| `ModuleNotFoundError: sktime_mcp` | Package not installed in the active environment | Confirm `.venv` is active (`which python`) and re-run `pip install sktime-mcp` |
| `pip: command not found` | System `pip` not available | Use `python -m pip` instead of bare `pip` |
| Wrong Python version selected | Multiple Python installations | Invoke `python3 -m venv .venv` explicitly and always use `python` inside the activated environment |

### Minimal Setup Check

After completing the steps above, confirm the server starts with `sktime-mcp`. See the [macOS / Unix-like first-time setup](#macos--unix-like-first-time-setup) section for the fallback command and common error solutions.

> **Note:** On Windows, the `sktime-mcp` command may be installed to a directory
> not on your `PATH` (e.g., `%APPDATA%\Python\Python3xx\Scripts`). Either add
> that directory to your `PATH` or use `python -m sktime_mcp.server` instead.


## 🚀 Quick Start

### Running the MCP Server
#### Standard Stdio Mode (for MCP Clients)
```bash
sktime-mcp
```

#### HTTP/SSE Mode via FastAPI (for Web Browsers or ChatGPT)
To expose the MCP server as a REST API over SSE (Server-Sent Events) for direct consumption:
```bash
PYTHONPATH=src .venv/bin/uvicorn sktime_mcp.app:app --host 127.0.0.1 --port 8001
```
This exposes standard SSE on `/sse` and message passing on `/messages/`.

> **Note for ChatGPT Web Users:** ChatGPT runs in the cloud and cannot connect to `http://127.0.0.1` (you will get an "Unsafe URL" error). You must expose your local server to the internet using a secure tunnel like [ngrok](https://ngrok.com/):
> ```bash
> ngrok http 8001
> ```
> Then use the provided `https://<your-ngrok-id>.ngrok-free.app/sse` URL in ChatGPT.

### Configuration (Environment Variables)

You can configure the server's behavior at runtime using environment variables:

- `SKTIME_MCP_MAX_RESPONSE_TOKENS`: Maximum tokens allowed per tool response (e.g., `10000`). If a response exceeds this limit, it is truncated and appended with a notice. Set to `0` (default) for unlimited.
- `SKTIME_MCP_LOG_LEVEL`: Server logging verbosity level (`DEBUG`, `INFO`, `WARNING`, `ERROR`). Defaults to `WARNING`.
- `SKTIME_MCP_AUTO_FORMAT`: Enables or disables automatic time-series formatting during data loading.
- `SKTIME_MCP_JOB_MAX_AGE_HOURS`: Maximum hours before completed background jobs are automatically pruned. Defaults to `24`.

### Connecting from an LLM Client

The server uses stdio transport by default, compatible with Claude Desktop, Claude Code, and other MCP clients.

**Claude Desktop** — add to your config file:

| Platform | Config path |
|----------|-------------|
| macOS    | `~/Library/Application Support/Claude/claude_desktop_config.json` |
| Linux    | `~/.config/claude/claude_desktop_config.json` |
| Windows  | `%APPDATA%\Claude\claude_desktop_config.json` |

**With uvx (recommended — no prior install needed):**
```json
{
  "mcpServers": {
    "sktime": {
      "command": "uvx",
      "args": ["sktime-mcp"]
    }
  }
}
```

**With optional extras:**
```json
{
  "mcpServers": {
    "sktime": {
      "command": "uvx",
      "args": ["sktime-mcp[forecasting,sql]"]
    }
  }
}
```

**With pip-installed package:**
```json
{
  "mcpServers": {
    "sktime": {
      "command": "sktime-mcp"
    }
  }
}
```

## ⚙️ Configuration

The server can be configured via environment variables:

| Environment Variable | Description | Default |
|----------------------|-------------|---------|
| `SKTIME_MCP_LOG_LEVEL` | Logging verbosity (e.g. `INFO`, `DEBUG`, `WARNING`) | `"WARNING"` |
| `SKTIME_MCP_LOG_PATH` | Optional file path to output logs to in addition to stderr | (None) |
| `SKTIME_MCP_AUTO_FORMAT` | Automatically format time series data on load (`true`/`false`) | `"true"` |
| `SKTIME_MCP_JOB_MAX_AGE_HOURS` | Maximum age in hours before background jobs are cleared | `24` |
| `SKTIME_MCP_JOB_CLEANUP_INTERVAL` | Interval in seconds for periodic job cleanup checks | `3600` |

## 📚 Available Tools

The full tool reference is in the project documentation: https://sktime.github.io/sktime-mcp/

| Need | Tool options | Rough explanation |
| --- | --- | --- |
| Discover what sktime can do | `list_available_data`, `query_registry`, `describe_component` | Find demo data, estimators, tags, and component details. |
| Bring data into the session | `load_data_source`, `inspect_data`, `transform_data`, `split_data`, `save_data` | Load files, inline data, SQL, or URLs into handles; inspect, clean, split, and persist them. |
| Build and run models | `instantiate_estimator`, `fit`, `predict`, `update`, `get_fitted_params`, `call_method` | Create sktime estimators or pipelines, fit them, forecast, update, or call native methods. |
| Evaluate and reproduce | `evaluate_estimator`, `export_code`, `save_model`, `load_model` | Cross-validate, generate Python code, and persist fitted models. |
| Manage runtime state | `list_handles`, `release_handle`, `release_data_handle`, `list_jobs`, `check_job_status`, `cancel_job` | See what is in memory, clean it up, and track async work. |

The practical mental model is simple: prompts create tool calls, tool calls create handles, and handles let later prompts continue the workflow.

## 🔄 Example LLM Flows

See the [User Guide](https://sktime.github.io/sktime-mcp/en/latest/user-guide.html) for end-to-end workflow examples, including:
- Discovering sktime coverage
- Retail forecasting and saving results
- Cleaning messy business data
- Time-series classification

## 📁 Project Structure

```
sktime-mcp/
├── src/sktime_mcp/
│   ├── server.py           # MCP server entry point
│   ├── registry/           # Registry interface & tag resolver
│   ├── composition/        # Pipeline composition validator
│   ├── runtime/            # Execution engine, handle & job management
│   ├── data/               # Data adapters (file, pandas, SQL, URL)
│   └── tools/              # MCP tool implementations
├── docs/                   # Sphinx documentation source
├── examples/               # Usage examples
├── tests/                  # Test suite
├── Dockerfile              # Multi-stage container build
├── docker-compose.yml      # Compose service definition
└── .dockerignore           # Docker build context filter
```

## 🧪 Running Tests

```bash
pytest tests/
```

## Local Quality Checks

Run standardized local checks before raising a PR:

```bash
make check
```

Auto-fix formatting and fixable lint issues:

```bash
make format-fix
```

If `make` is unavailable (common on Windows), run the equivalent commands:

```bash
ruff format --check .
ruff check .
pytest
```

### Pre-Commit Hooks (Recommended)
To ensure your code meets quality standards before pushing, install the pre-commit hooks:
```bash
make install-hooks
```
This will automatically run Ruff and Pytest on your code every time you make a commit.