"""Tests for demo dataset loading semantics in the executor."""

import sys

sys.path.insert(0, "src")


def test_load_dataset_keeps_forecasting_demo_supported():
    """Forecasting demo datasets should still load successfully."""
    from sktime_mcp.runtime.executor import Executor

    executor = Executor()
    result = executor.load_dataset("airline")

    assert result["success"] is True
    assert "data" in result


def test_load_dataset_rejects_non_forecasting_supervised_demo():
    """Auto-discovered supervised demo datasets should fail fast with a clear error."""
    from sktime_mcp.runtime.executor import DEMO_DATASETS, Executor

    assert "basic_motions" in DEMO_DATASETS

    executor = Executor()
    result = executor.load_dataset("basic_motions")

    assert result["success"] is False
    assert "supervised/non-forecasting" in result["error"]
    assert "(X, y)" in result["error"]
