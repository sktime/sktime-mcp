import importlib.util
from typing import Any

import pytest

# Attempt to detect sktime for CI vs local dev environments
_SKTIME_AVAILABLE = importlib.util.find_spec("sktime") is not None
if not _SKTIME_AVAILABLE:
    pytest.skip("sktime is not installed", allow_module_level=True)

from sktime.forecasting.naive import NaiveForecaster

from sktime_mcp.runtime.executor import Executor, get_executor
from sktime_mcp.runtime.handles import HandleManager, get_handle_manager
from sktime_mcp.tools.tune import tune_estimator_tool


@pytest.fixture(autouse=True)
def cleanup_executor():
    """Reset the executor and handle manager state between tests."""
    # Reset singleton instances
    import sktime_mcp.runtime.executor as executor_module
    import sktime_mcp.runtime.handles as handles_module

    # Save original
    old_executor = executor_module._executor_instance
    old_handler = handles_module._handle_manager_instance

    # Clear
    executor_module._executor_instance = None
    handles_module._handle_manager_instance = None

    yield

    # Restore (optional, but good practice in case other tests run alongside)
    executor_module._executor_instance = old_executor
    handles_module._handle_manager_instance = old_handler


def test_tune_estimator_tool_missing_handle():
    """Test tuning fails gracefully if the handle does not exist."""
    # Attempt to tune a non-existent handle
    result = tune_estimator_tool(
        estimator_handle="non_existent",
        dataset="airline",
        param_grid={"sp": [2, 4]},
    )
    assert not result["success"]
    assert "Handle not found" in result["error"]


def test_tune_estimator_tool_success():
    """Test successful grid search tuning."""
    # 1. Setup execution environment
    executor = get_executor()
    handle_manager = get_handle_manager()

    # 2. Register naive forecaster
    instance = NaiveForecaster()
    handle_id = handle_manager.create_handle("NaiveForecaster", instance)

    # 3. Call tuning tool
    # We pass a simple parameter grid
    param_grid = {"strategy": ["last", "mean", "drift"]}

    result = tune_estimator_tool(
        estimator_handle=handle_id,
        dataset="airline",
        param_grid=param_grid,
        cv_folds=2,
    )

    # 4. Verify results
    assert result["success"] is True
    assert "best_params" in result
    assert "strategy" in result["best_params"]
    assert "best_score" in result
    assert isinstance(result["best_score"], float)

    # 5. Verify the new handle was registered
    tuned_handle = result["tuned_handle"]
    assert tuned_handle.startswith("NaiveForecaster_tuned")

    # Ensure it's marked as fitted
    handle_info = handle_manager.get_info(tuned_handle)
    assert handle_info.fitted is True

    tuned_instance = handle_manager.get_instance(tuned_handle)
    assert tuned_instance.__class__.__name__ == "NaiveForecaster"
    # Should automatically have best parameter populated
    assert tuned_instance.strategy == result["best_params"]["strategy"]
