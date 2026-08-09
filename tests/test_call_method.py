"""Tests for Executor.call_method."""

import json

import pytest

from sktime_mcp.runtime.executor import get_executor


class TestCallMethodSplitters:
    """Splitter generator results are returned as fold lists, not str(gen)."""

    def test_split_returns_folds(self):
        """SlidingWindowSplitter.split with y_dataset=airline (issue #490)."""
        executor = get_executor()
        inst = executor.instantiate(
            "SlidingWindowSplitter(window_length=36, step_length=12, fh=[1, 2, 3])"
        )
        assert inst["success"], inst

        out = executor.call_method(inst["handle"], "split", {"y_dataset": "airline"})
        assert out["success"] is True, out
        assert isinstance(out["result"], list)
        assert len(out["result"]) >= 1
        train, test = out["result"][0]
        assert isinstance(train, list) and isinstance(test, list)
        assert all(isinstance(i, int) for i in train + test)
        assert len(test) == 3
        json.dumps(out)

    def test_split_matches_native(self):
        from sktime.datasets import load_airline
        from sktime.split import SlidingWindowSplitter

        y = load_airline()
        cv = SlidingWindowSplitter(window_length=36, step_length=12, fh=[1, 2, 3])
        native = list(cv.split(y))

        executor = get_executor()
        inst = executor.instantiate(
            "SlidingWindowSplitter(window_length=36, step_length=12, fh=[1, 2, 3])"
        )
        out = executor.call_method(inst["handle"], "split", {"y_dataset": "airline"})
        assert out["success"] is True, out
        assert len(out["result"]) == len(native)
        for got, (tr, te) in zip(out["result"], native, strict=True):
            assert got[0] == tr.tolist()
            assert got[1] == te.tolist()

    @pytest.mark.parametrize("method", ["split", "split_loc", "split_series"])
    def test_generator_methods(self, method):
        executor = get_executor()
        inst = executor.instantiate("SlidingWindowSplitter(window_length=24, step_length=12, fh=3)")
        assert inst["success"], inst
        out = executor.call_method(inst["handle"], method, {"y_dataset": "airline"})
        assert out["success"] is True, out
        assert isinstance(out["result"], list)
        assert len(out["result"]) >= 1
        json.dumps(out)

    def test_unknown_handle(self):
        out = get_executor().call_method("est_missing", "split", {})
        assert out["success"] is False
