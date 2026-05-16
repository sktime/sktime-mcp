"""Smoke tests for pipeline and own-data workflow examples."""

import importlib.util
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = REPO_ROOT / "examples"


def _run_example(name: str) -> str:
    """Run an example script and return stdout."""
    result = subprocess.run(
        [sys.executable, str(EXAMPLES_DIR / name)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout


def test_pipeline_instantiation_example_runs_successfully():
    """The pipeline-instantiation example should remain runnable."""
    output = _run_example("03_pipeline_instantiation.py")

    assert "All examples completed!" in output
    assert "Success: True" in output
    assert "Predictions:" in output
    assert "Validation errors:" in output


def test_pandas_example_runs_successfully():
    """The in-memory own-data example should continue to work end to end."""
    output = _run_example("pandas_example.py")

    assert "Method 1: Using DataSourceRegistry" in output
    assert "Method 2: Using Executor" in output
    assert "Predictions: True" in output
    assert "Cleanup:" in output


def test_csv_example_runs_successfully():
    """The CSV/TSV own-data example should remain runnable."""
    output = _run_example("csv_example.py")

    assert "Loading data from CSV file" in output
    assert "Loading data from TSV file" in output
    assert "Forecast for next 10 days:" in output
    assert "Example completed!" in output


def test_sql_example_handles_optional_dependency_cleanly():
    """The SQL example should either run or explain the missing optional dependency clearly."""
    output = _run_example("sql_example.py")

    if importlib.util.find_spec("sqlalchemy") is None:
        assert "Optional dependency missing: SQLAlchemy is not installed." in output
        assert "Skipping runnable SQLite examples" in output
    else:
        assert "Example 1: Load with SQL query" in output
        assert "Load result: True" in output

    assert "Example completed!" in output
