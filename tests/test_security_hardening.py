"""Conservative trust-boundary hardening (#540).

Covers the non-controversial parts: block private/dunder methods in
call_method (BUG-11), reject non-estimator instantiate results (BUG-10), and
cap run_command output (BUG-23). instantiate sandboxing (BUG-09) is
intentionally out of scope — the server already exposes run_command.
"""

import contextlib

import pytest

from sktime_mcp.runtime.executor import get_executor
from sktime_mcp.runtime.handles import get_handle_manager
from sktime_mcp.tools.instantiate import instantiate_tool
from sktime_mcp.tools.run_command import run_command_tool


def _release(handle):
    with contextlib.suppress(KeyError):
        get_handle_manager().release_handle(handle)


class TestCallMethodDenylist:
    def test_reduce_blocked(self):
        h = instantiate_tool(spec="NaiveForecaster()")["handle"]
        try:
            res = get_executor().call_method(handle_id=h, method_name="__reduce__", kwargs={})
            assert not res["success"]
            assert "private" in res["error"].lower()
        finally:
            _release(h)

    @pytest.mark.parametrize("m", ["__class__", "__getattribute__", "_get_class_flags", "__init__"])
    def test_private_methods_blocked(self, m):
        h = instantiate_tool(spec="NaiveForecaster()")["handle"]
        try:
            res = get_executor().call_method(handle_id=h, method_name=m, kwargs={})
            assert not res["success"]
            assert "private" in res["error"].lower()
        finally:
            _release(h)

    def test_public_method_still_works(self):
        h = instantiate_tool(spec="NaiveForecaster(sp=12)")["handle"]
        try:
            res = get_executor().call_method(handle_id=h, method_name="get_params", kwargs={})
            assert res["success"]
            assert res["result"]["sp"] == 12
        finally:
            _release(h)


class TestInstantiateTypeCheck:
    @pytest.mark.parametrize("spec", ["42", "[1, 2, 3]", "'just a string'", "None"])
    def test_non_estimator_rejected(self, spec):
        res = instantiate_tool(spec=spec)
        assert not res["success"]
        assert "did not produce an sktime estimator" in res["error"]

    def test_real_estimator_still_instantiates(self):
        res = instantiate_tool(spec="NaiveForecaster(sp=12)")
        try:
            assert res["success"]
        finally:
            _release(res.get("handle"))


class TestRunCommandCap:
    def test_large_output_truncated(self):
        res = run_command_tool("seq 1 100000")
        assert res["success"]
        assert res["truncated"] is True
        assert len(res["output"]) < 25_000
        assert "truncated" in res["output"]

    def test_small_output_not_truncated(self):
        res = run_command_tool("echo hello")
        assert res["success"]
        assert res["truncated"] is False
        assert res["output"] == "hello"

    def test_nonzero_exit_has_error_key(self):
        res = run_command_tool("exit 3")
        assert not res["success"]
        assert res["returncode"] == 3
        assert "error" in res
