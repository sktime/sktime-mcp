# Installation

This guide covers the steps required to install `sktime-mcp` for both end-users and developers.

## Prerequisites

- **Python**: Version 3.10 or higher.
- **pip**: The Python package installer.
- **MCP Client**: A compatible Model Context Protocol client such as Claude Desktop, Cursor, or a VS Code extension (e.g., Cline).

## Standard Installation

The recommended way to install `sktime-mcp` is via pip or uv. To enable full functionality, including support for various data formats and forecasting models, use the `[all]` extra.

### Using pip
```bash
pip install "sktime-mcp[all]"
```

### Using uv
If you prefer [uv](https://github.com/astral-sh/uv), you can run the server directly without manual installation using `uvx`:

```bash
uvx sktime-mcp
```

Or install it globally:
```bash
uv tool install "sktime-mcp[all]"
```

### Optional Extras

Depending on your use case, you may choose to install specific dependency groups:

| Extra | Description |
| :--- | :--- |
| `forecasting` | Extended forecasting models (Prophet, TBATS, StatsForecast). |
| `sql` | Support for loading data from SQL databases (PostgreSQL, MySQL). |
| `files` | Support for Excel and Parquet file formats. |
| `mlflow` | Integration with MLflow for model persistence. |
| `dl` | Deep learning models (TensorFlow, PyTorch). |

To install a specific extra, use:
```bash
pip install "sktime-mcp[extra_name]"
```

## Developer Installation

If you intend to contribute to the project or run the test suite, install `sktime-mcp` from source in editable mode.

1. **Clone the repository**:
   ```bash
   git clone https://github.com/sktime/sktime-mcp.git
   cd sktime-mcp
   ```

2. **Create and activate a virtual environment**:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
   ```

3. **Install in editable mode with development dependencies**:
   ```bash
   pip install -e ".[dev,all]"
   ```

## Verification

After installation, verify that the `sktime-mcp` CLI tool is available by checking its version:

```bash
sktime-mcp --version
```

If the command is not found, ensure that your Python scripts directory is in your system's PATH.
