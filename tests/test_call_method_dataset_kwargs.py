"""call_method must surface *_dataset kwargs that fail to load.

Previously an unresolvable dataset name was silently left in kwargs, so
the raw ``y_dataset`` string reached the method and produced a confusing
"unexpected keyword argument 'y_dataset'" error instead of the load
failure.
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


def test_unknown_dataset_returns_load_error(splitter_handle):
    result = get_executor().call_method(
        handle_id=splitter_handle,
        method_name="get_n_splits",
        kwargs={"y_dataset": "no_such_dataset_zzz"},
    )
    assert result["success"] is False
    assert "Unknown dataset: no_such_dataset_zzz" in result["error"]
    assert "unexpected keyword argument" not in result["error"]
    assert "available" in result


def test_known_dataset_still_resolves(splitter_handle):
    result = get_executor().call_method(
        handle_id=splitter_handle,
        method_name="get_n_splits",
        kwargs={"y_dataset": "airline"},
    )
    assert result["success"], result
    assert isinstance(result["result"], int)
