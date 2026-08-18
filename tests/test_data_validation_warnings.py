"""Tests for data validation warning propagation."""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sktime_mcp.runtime.executor import Executor


def _ambiguous_target_config():
    return {
        "type": "pandas",
        "data": {
            "date": ["2020-01-01", "2020-01-02", "2020-01-03"],
            "value": [1, 2, 3],
        },
    }


def test_load_data_source_propagates_default_target_warning():
    """Default target warnings should appear in the top-level validation result."""
    result = Executor().load_data_source(_ambiguous_target_config())

    assert result["success"] is True
    warnings = result["validation"]["warnings"]
    assert any("Target column not specified" in warning for warning in warnings)


def test_load_data_source_async_propagates_default_target_warning():
    """Async load should expose the same default-target warning as sync load."""
    result = asyncio.run(Executor().load_data_source_async(_ambiguous_target_config()))

    assert result["success"] is True
    warnings = result["validation"]["warnings"]
    assert any("Target column not specified" in warning for warning in warnings)

