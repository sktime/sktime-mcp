"""Smoke test for the external agent integration example."""

import sys
from pathlib import Path
from subprocess import run


def test_external_agent_integration_example_runs():
    """The example should run and print key integration guidance."""
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "examples" / "08_external_agent_integration.py"

    result = run(
        [sys.executable, str(script)],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    )

    stdout = result.stdout
    assert "External Agent Integration Workflow" in stdout
    assert "Use editable install first: pip install -e ." in stdout
    assert "Fallback during source-only integration: PYTHONPATH=/path/to/sktime-mcp/src" in stdout
    assert "Metadata summary:" in stdout
    assert "Custom-data forecast horizon: 3" in stdout
    assert "Evaluation folds requested: 3" in stdout
    assert "Returned metric keys:" in stdout
    assert "External agent workflow complete!" in stdout
