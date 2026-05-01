"""
Smoke tests for example scripts.

Ensures that all scripts in the examples directory run without crashing.
This prevents API drift where examples become outdated relative to the main codebase.
"""

import os
import runpy
from pathlib import Path

import pytest

# Get the directory containing example scripts
EXAMPLES_DIR = Path(__file__).parent.parent / "examples"

# Get all .py files in the examples directory
EXAMPLE_SCRIPTS = [
    f
    for f in os.listdir(EXAMPLES_DIR)
    if f.endswith(".py") and f != "__init__.py"
]


@pytest.mark.parametrize("script_name", EXAMPLE_SCRIPTS)
def test_example_script(script_name):
    """
    Run an example script and ensure it completes without unhandled exceptions.
    """
    script_path = EXAMPLES_DIR / script_name

    # We use runpy to execute the script in its own namespace
    try:
        runpy.run_path(str(script_path), run_name="__main__")
    except Exception as e:
        pytest.fail(f"Example script '{script_name}' failed with exception: {e}")
