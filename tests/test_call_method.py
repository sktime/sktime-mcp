"""call_method JSON sanitization beyond generator materialization."""

import json

import pandas as pd
from sktime.datasets import load_airline

from sktime_mcp.runtime.executor import get_executor
from sktime_mcp.tools.instantiate import instantiate_tool


def test_split_loc_returns_label_lists():
    """split_loc folds are label lists, not PeriodIndex repr strings."""
    executor = get_executor()
    inst = instantiate_tool(spec="SlidingWindowSplitter(window_length=24, step_length=12)")
    assert inst["success"], inst
    handle = inst["handle"]
    try:
        out = executor.call_method(handle, "split_loc", {"y_dataset": "airline"})
        assert out["success"] is True, out
        train, test = out["result"][0]
        assert isinstance(train, list) and isinstance(test, list)
        assert train and all(isinstance(x, str) for x in train + test)
        json.dumps(out)
    finally:
        executor._handle_manager.release_handle(handle)


def test_to_dict_path_is_json_safe():
    """Series.to_dict() still goes through sanitize_for_json (PeriodIndex keys)."""
    executor = get_executor()
    inst = instantiate_tool(spec="NaiveForecaster()")
    handle = inst["handle"]
    try:
        fit_res = executor.fit(handle, y=load_airline())
        assert fit_res["success"], fit_res
        out = executor.call_method(handle, "predict", {"fh": [1, 2, 3]})
        assert out["success"] is True, out
        json.dumps(out)
    finally:
        executor._handle_manager.release_handle(handle)


def test_multiindex_flatten_copies_frame():
    """Flattening MultiIndex columns must not mutate the live object."""
    df = pd.DataFrame(
        [[1.0, 2.0]],
        columns=pd.MultiIndex.from_tuples([("Coverage", 0.9), ("Coverage", 0.1)]),
    )

    class _Frame:
        def dump(self):
            return df

    executor = get_executor()
    handle = executor._handle_manager.create_handle("Frame", _Frame(), {})
    try:
        out = executor.call_method(handle, "dump", {})
        assert out["success"] is True, out
        assert isinstance(df.columns, pd.MultiIndex)
        json.dumps(out)
    finally:
        executor._handle_manager.release_handle(handle)
