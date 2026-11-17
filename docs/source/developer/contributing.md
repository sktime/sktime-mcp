# Contributing Guide

Thank you for your interest in contributing to `sktime-mcp`. This project aims to provide a robust MCP layer for the `sktime` ecosystem.

## Development Setup

1. **Fork and Clone**:
   ```bash
   git clone https://github.com/sktime/sktime-mcp.git
   cd sktime-mcp
   ```

2. **Environment**:
   We recommend using a virtual environment.
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install Dependencies**:
   Install the package in editable mode with all development and optional dependencies.
   ```bash
   pip install -e ".[dev,all]"
   ```

4. **Pre-commit Hooks**:
   Enable pre-commit hooks to ensure code quality and formatting.
   ```bash
   pre-commit install
   ```

## Code Quality Standards

We use `ruff` for linting and formatting.

- **Check Linting**: `ruff check .`
- **Format Code**: `ruff format .`

Ensure all new code is properly type-hinted and includes docstrings following the NumPy format.

## Testing

The project uses `pytest`. All new features should be accompanied by unit or integration tests.

### Running Tests
```bash
pytest tests/
```

### Async Tests
Since the MCP server is heavily asynchronous, many tests use `pytest-asyncio`. Ensure your test environment is correctly configured.

## Project Structure

- `src/sktime_mcp/server.py`: The main entry point, tool schemas, and tool dispatcher.
- `src/sktime_mcp/tools/`: Implementation of individual MCP tools, grouped by category.
- `src/sktime_mcp/runtime/`: State management, handle registry, and job execution.
- `src/sktime_mcp/registry/`: sktime registry introspection and tag resolution.
- `src/sktime_mcp/data/`: Data source adapters, loading, and formatting logic.

Pipeline composition has no module of its own — it is expressed in the
`instantiate` spec string and validated by sktime's `craft()`.

## Documentation

Docs live in `docs/source/` and are built with Sphinx + MyST.

```bash
sphinx-build -W docs/source docs/_build/html
```

`-W` turns warnings into errors, which is what CI enforces — **the build must stay
warning-free**. If you add or change a tool, update `docs/source/tool-reference.md`
in the same PR.

When writing NumPy-style docstrings, a bullet list needs a blank line before it,
and each bullet's description must stay on the bullet's own line or be indented to
align with the bullet text. Otherwise docutils reports "Unexpected indentation"
and the section renders incorrectly in the API reference.

## Pull Request Process

1. Create a new branch for your feature or bug fix.
2. Ensure all tests pass and linting is clean.
3. Update the documentation if you are adding or changing tools.
4. Submit a PR with a clear description of the changes and references to any related issues.
