"""Smoke tests for documented agentic/MCP workflow examples."""

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


def test_forecasting_workflow_example_runs_successfully():
    """The end-to-end forecasting workflow example should finish cleanly."""
    output = _run_example("01_forecasting_workflow.py")

    assert "Demo Complete" in output
    assert "Valid: False" in output
    assert "Error:" not in output


def test_llm_query_simulation_example_runs_successfully():
    """The LLM-style query simulation should not hide failed workflow steps."""
    output = _run_example("02_llm_query_simulation.py")

    assert "All LLM Query Simulations Complete" in output
    assert "ThetaForecaster predictions generated successfully" in output
    assert "NaiveForecaster predictions generated successfully" in output
    assert '"success": false' not in output
    assert "Unknown estimator: Detrend" not in output


def test_pipeline_demo_example_runs_successfully():
    """The pipeline demo should continue to show a successful two-call workflow."""
    output = _run_example("04_mcp_pipeline_demo.py")

    assert "SUCCESS! LLM created and used a complete pipeline" in output
    assert '"success": true' in output
