"""call_method must materialize generator results.

Splitter methods like split/split_loc return generators; without
materialization the caller gets a useless repr string such as
"<generator object BaseSplitter.split at 0x...>".
"""

import contextlib

import pytest

from sktime_mcp.runtime.executor import get_executor
from sktime_mcp.runtime.handles import get_handle_manager
from sktime_mcp.tools.instantiate import instantiate_tool


@pytest.fixture
def splitter_handle():
    result = instantiate_tool(spec="SlidingWindowSplitter(window_length=24, step_length=12)")
    assert result["success"], f"Failed to create handle: {result}"
    yield result["handle"]
    with contextlib.suppress(KeyError):
        get_handle_manager().release_handle(result["handle"])


def test_split_returns_materialized_folds(splitter_handle):
    result = get_executor().call_method(
        handle_id=splitter_handle,
        method_name="split",
        kwargs={"y_dataset": "airline"},
    )
    assert result["success"], result
    folds = result["result"]
    assert isinstance(folds, list)
    assert len(folds) > 0
    assert "generator object" not in str(folds)
    # Each fold is a (train_indices, test_indices) pair of JSON-safe int lists
    train, test = folds[0]
    assert all(isinstance(i, int) for i in train)
    assert all(isinstance(i, int) for i in test)
    assert len(train) == 24


def test_non_generator_results_unchanged(splitter_handle):
    result = get_executor().call_method(
        handle_id=splitter_handle,
        method_name="get_n_splits",
        kwargs={"y_dataset": "airline"},
    )
    assert result["success"], result
    assert isinstance(result["result"], int)
