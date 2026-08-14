"""Smoke test for the agentic model-selection workflow example."""

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_PATH = REPO_ROOT / "examples" / "07_agentic_model_selection.py"


def test_agentic_model_selection_example_runs_successfully():
    """The agentic workflow example should complete end to end."""
    result = subprocess.run(
        [sys.executable, str(EXAMPLE_PATH)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )

    output = result.stdout

    assert "Agentic Model-Selection Workflow Demo" in output
    assert "Shortlisted candidates:" in output
    assert "Leaderboard (lower mean MAPE is better):" in output
    assert "Selected winner:" in output
    assert "Forecast generated for 12 steps." in output
    assert "Exported code preview:" in output
    assert "Agentic workflow complete!" in output
